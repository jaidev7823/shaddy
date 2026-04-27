import json, os, wave, tempfile, threading, subprocess, time
import numpy as np
import pyaudio, webrtcvad
from faster_whisper import WhisperModel
from pathlib import Path
import requests
import torch
import soundfile as sf
from chatterbox.tts_turbo import ChatterboxTurboTTS
from queue import Queue, Empty

BASE   = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Models ────────────────────────────────────────────────────────────────────
tts_model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
whisper   = WhisperModel("medium", device="cuda", compute_type="float16")
VOICE_REF = str(BASE / "l_voice_sample.wav")

# ── Audio config ──────────────────────────────────────────────────────────────
MIC_RATE    = 16000
VAD_RATE    = 16000
FRAME_MS    = 30
FRAME_BYTES = int(VAD_RATE * FRAME_MS / 1000) * 2
DOWNSAMPLE  = MIC_RATE // VAD_RATE

# ── Lessons ───────────────────────────────────────────────────────────────────
lessons = json.loads((BASE / "lessons.json").read_text())

# ── Cooldown ──────────────────────────────────────────────────────────────────
COOLDOWN  = 10
cooldowns = {}

# ── Single worker queue — no parallel LLM/TTS calls ever ─────────────────────
audio_queue = Queue(maxsize=1)  # maxsize=1 drops stale audio if worker is busy

# ── Prompt ───────────────────────────────────────────────────────────────────
def build_prompt(transcript: str) -> str:
    topics = "\n".join(f'{l["id"]}: {l["topic"]}' for l in lessons)
    return f"""### Role: English Teacher
The student is a beginner. Provide a direct sentence for them to repeat.

### Topics:
{topics}

### Input:
"{transcript}"

### Task:
If the input allows for a grammar-based reply, provide:
1. A direct answer (the sentence they should say).
2. The lesson_id.
3. A 3-word grammar concept.

If no specific grammar applies, still provide a simple "answer" so the student isn't left in silence.

### JSON Format:
{{
  "lesson_id": "L101",
  "answer": "Yes, I am happy.",
  "concept": "Present Tense"
}}"""

# ── LLM ──────────────────────────────────────────────────────────────────────
def ask_llm(transcript: str) -> dict:
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2:0.5b",
                "prompt": build_prompt(transcript),
                "format": "json",
                "stream": False
            },
            timeout=15
        )
        data = json.loads(r.json()["response"])
        
        # If there is an answer, we SHOULD nudge/speak
        answer = data.get("answer", "")
        should_nudge = len(answer) > 2
        
        return {
            "should_nudge": should_nudge,
            "lesson_id": data.get("lesson_id"),
            "answer": answer,
            "concept": data.get("concept", "English Basics")
        }
    except Exception:
        return {"should_nudge": False}

# ── TTS ───────────────────────────────────────────────────────────────────────
def generate_and_play(text: str):
    try:
        with torch.no_grad():
            wav = tts_model.generate(text, audio_prompt_path=VOICE_REF)
        wav = wav.detach().cpu().contiguous()
        out_path = BASE / "generated.wav"
        sf.write(str(out_path), wav.squeeze().numpy(), tts_model.sr)
        # wait for playback to finish before returning
        subprocess.run(["aplay", "-q", str(out_path)])
        del wav
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  TTS error: {e}")

# ── Worker — runs in one dedicated thread, processes one chunk at a time ──────
def worker():
    while True:
        try:
            audio = audio_queue.get(timeout=1)
        except Empty:
            continue

        tmp = None
        try:
            # 1. Write wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2)
                    w.setframerate(MIC_RATE); w.writeframes(audio)
                tmp = f.name

            # 2. Transcribe
            segs, _ = whisper.transcribe(tmp, language="hi", vad_filter=True, task="translate")
            text = " ".join(s.text for s in segs).strip()
            os.unlink(tmp); tmp = None

            if not text:
                continue
            print(f'  heard: "{text}"')

            # 3. LLM decides
            result = ask_llm(text)
            print(f"  llm: {result}")

            if not result.get("should_nudge"):
                continue

            lesson_id = result.get("lesson_id")
            nudge     = result.get("nudge")

            if not nudge:
                continue

            # 4. Cooldown
            now = time.time()
            if lesson_id and (now - cooldowns.get(lesson_id, 0)) < COOLDOWN:
                print(f"  skipped (cooldown): {lesson_id}")
                continue

            # 5. Speak
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

# ── Mic loop ──────────────────────────────────────────────────────────────────
def main():
    # start single worker thread
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    vad    = webrtcvad.Vad(3)
    pa     = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=1,
        rate=MIC_RATE, input=True,
        frames_per_buffer=int(MIC_RATE * FRAME_MS / 1000)
    )
    print("Listening... (Ctrl+C to stop)")

    buf, speech, silence, active = [], 0, 0, False
    try:
        while True:
            raw   = stream.read(int(MIC_RATE * FRAME_MS / 1000), exception_on_overflow=False)
            arr   = np.frombuffer(raw, dtype=np.int16)
            arr16 = arr[::DOWNSAMPLE].astype(np.int16)
            r16   = arr16.tobytes()[:FRAME_BYTES].ljust(FRAME_BYTES, b'\x00')

            if vad.is_speech(r16, VAD_RATE):
                buf.append(raw); speech += 1; silence = 0; active = True
            elif active:
                buf.append(raw); silence += 1
                if silence > 300 // FRAME_MS:
                    if speech > 12:
                        chunk = b"".join(buf)
                        # non-blocking put — drops chunk if worker is busy
                        # prevents queue buildup while TTS is playing
                        try:
                            audio_queue.put_nowait(chunk)
                        except:
                            print("  (busy, dropped chunk)")
                    buf, speech, silence, active = [], 0, 0, False

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

main()
