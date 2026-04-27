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
from google import genai
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# This looks for the .env file and loads it into os.environ
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)

class ESLResponse(BaseModel):
    lesson_id: str
    answer: str
    hint: str

BASE   = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Models ────────────────────────────────────────────────────────────────────
tts_model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
whisper = WhisperModel("medium", device="cuda", compute_type="float16")
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
def build_prompt(transcript: str, speaker: str = "student") -> str:
    # speaker = "student" → student said this, correct it
    # speaker = "other"   → other person said this, help student reply
    topics = "\n".join(f'{l["id"]}: {l["topic"]}' for l in lessons)

    if speaker == "student":
        situation = f"""The student just said this to their conversation partner:
"{transcript}"
Your job: Rewrite it as a more natural, fluent English sentence they can say right now."""

    else:  # speaker == "other"
        situation = f"""The other person just said this TO the student:
"{transcript}"
The student needs to reply. Your job: Give the student a natural, fluent English response they can say right now."""

    return f"""
### MISSION
You are a "Shadow Assistant" for an ESL student. You are silently observing their conversation.

### THE SITUATION
- You are NOT part of the conversation.
- Do NOT respond to the meaning for yourself.
- Your ONLY job: give the student the next sentence to speak.

### AVAILABLE TOPICS
{topics}

### CURRENT MOMENT
{situation}

### YOUR TASK
1. **answer** — A natural English sentence the student can immediately say aloud. No explanations. No grammar terms.
2. **hint** — One short phrase (max 8 words) naming the grammar concept. Example: "Uses simple past for completed actions."
3. **lesson_id** — Most relevant topic ID. Use "GEN-01" if none fit.

### STRICT RULES
- "answer" is always something the student SPEAKS to their partner.
- "answer" NEVER contains grammar notes, brackets, or meta-comments.
- "hint" is one concept label only.

### OUTPUT — valid JSON only:
{{
  "lesson_id": "string",
  "answer": "string",
  "hint": "string"
}}"""

# ── LLM ──────────────────────────────────────────────────────────────────────
def ask_llm(transcript: str, speaker: str = "student") -> dict:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=build_prompt(transcript, speaker),
            config={
                "system_instruction": "You are a silent ESL coach. Output JSON only.",
                "response_mime_type": "application/json",
                "response_schema": ESLResponse,
                "temperature": 0.1,
            }
        )

        parsed = response.parsed
        nudge = parsed.answer
        should_nudge = bool(nudge and nudge.upper() != "NONE" and len(nudge) > 2)

        return {
            "should_nudge": should_nudge,
            "lesson_id": parsed.lesson_id,
            "nudge": parsed.answer,
            "hint": parsed.hint
        }
    except Exception as e:
        print(f"  Gemini Error: {e}")
        return {"should_nudge": False, "lesson_id": None, "nudge": None}

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
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(MIC_RATE)
                    w.writeframes(audio)
                tmp = f.name

            # 2. Transcribe (Optimized for Hindi + Accuracy)
            segs, info = whisper.transcribe(
                tmp,
                language="en",
                task="translate",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),

                # --- ACCURACY TWEAKS ---
                temperature=0.0,
                beam_size=5,
                # --- NOISE FILTER ---
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4
            )

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
