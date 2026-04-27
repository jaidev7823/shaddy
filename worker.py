import os
import tempfile
import time
import wave
from queue import Queue, Empty

import webrtcvad

from config import (
    MIC_RATE,
    VAD_RATE,
    FRAME_MS,
    FRAME_BYTES,
    DOWNSAMPLE,
    COOLDOWN,
)
from models import whisper
from prompt import build_prompt  # noqa: F401
from llm import ask_llm
from audio import generate_and_play

cooldowns = {}
audio_queue = Queue(maxsize=1)


def worker():
    while True:
        try:
            audio = audio_queue.get(timeout=1)
        except Empty:
            continue

        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(MIC_RATE)
                    w.writeframes(audio)
                tmp = f.name

            segs, info = whisper.transcribe(
                tmp,
                language="en",
                task="translate",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                temperature=0.0,
                beam_size=5,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4
            )

            text = " ".join(s.text for s in segs).strip()
            os.unlink(tmp); tmp = None

            if not text:
                continue
            print(f'  heard: "{text}"')

            result = ask_llm(text)
            print(f"  llm: {result}")

            if not result.get("should_nudge"):
                continue

            lesson_id = result.get("lesson_id")
            nudge = result.get("nudge")

            if not nudge:
                continue

            now = time.time()
            if lesson_id and (now - cooldowns.get(lesson_id, 0)) < COOLDOWN:
                print(f"  skipped (cooldown): {lesson_id}")
                continue

            print(f"  → nudge: {nudge}")
            generate_and_play(nudge)

            if lesson_id:
                cooldowns[lesson_id] = now

        except Exception as e:
            print(f"  Worker error: {e}")
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        finally:
            audio_queue.task_done()


def get_queue():
    return audio_queue