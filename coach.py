#!/usr/bin/env python3
import json, os, wave, tempfile, threading, subprocess, time, datetime
import numpy as np
import pyaudio, webrtcvad
from faster_whisper import WhisperModel
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MIC_SAMPLE_RATE = 16000       # change if your mic differs
VAD_RATE        = 16000       # webrtcvad needs 16k
FRAME_MS        = 30          # 10/20/30 only
FRAME_BYTES     = int(VAD_RATE * FRAME_MS / 1000) * 2
COOLDOWN        = 5          # seconds before same nudge repeats
DOWNSAMPLE      = MIC_SAMPLE_RATE // VAD_RATE  # 44100//16000 = 2
# ─────────────────────────────────────────────────────────────────────────────

BASE    = Path(__file__).parent
lessons = json.loads((BASE / "lessons.json").read_text())
model = WhisperModel("medium", device="cpu", compute_type="int8")
cooldowns = {}

def play(lesson_id):
    wav = BASE / "nudges" / f"{lesson_id}.wav"
    if wav.exists():
        subprocess.Popen(["aplay", "-q", str(wav)])

def prebake():
    (BASE / "nudges").mkdir(exist_ok=True)
    for l in lessons:
        out = BASE / "nudges" / f"{l['id']}.wav"
        if not out.exists():
            subprocess.run(["espeak-ng", "-w", str(out), "-s", "130", "-p", "40", l["nudge"]])

def on_speech(audio: bytes):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(MIC_SAMPLE_RATE)
                w.writeframes(audio)
            tmp = f.name

        # Add language="en" (or your lang) to bypass the failing auto-detection
        segs, info = model.transcribe(tmp, language="hi", vad_filter=True, task="translate")
        
        text = " ".join(s.text for s in segs).strip().lower()
        os.unlink(tmp)
        
        if not text: return
        print(f"  \"{text}\"")

        now = time.time()
        for l in lessons:
            if now - cooldowns.get(l["id"], 0) < COOLDOWN: continue
            if any(kw.lower() in text for kw in l["keywords"]):
                print(f"  → {l['nudge']}")
                play(l["id"])
                cooldowns[l["id"]] = now
                with open(BASE / "session.log", "a") as f:
                    f.write(json.dumps({"t": datetime.datetime.now().isoformat(),
                        "lesson": l["id"], "heard": text}) + "\n")

    except Exception as e:
        print(f"Transcription error: {e}")
        if os.path.exists(tmp): os.unlink(tmp)

def main():
    prebake()
    vad = webrtcvad.Vad(2)
    pa  = pyaudio.PyAudio()

    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=MIC_SAMPLE_RATE,
                     input=True, frames_per_buffer=int(MIC_SAMPLE_RATE * FRAME_MS / 1000))

    print("Listening... (Ctrl+C to stop)")
    buf, speech, silence, active = [], 0, 0, False

    try:
        while True:
            raw = stream.read(int(MIC_SAMPLE_RATE * FRAME_MS / 1000), exception_on_overflow=False)
            # downsample to 16k for VAD (audioop removed in Python 3.13+)
            arr = np.frombuffer(raw, dtype=np.int16)
            arr16 = arr[::DOWNSAMPLE].astype(np.int16)
            r16 = arr16.tobytes()[:FRAME_BYTES].ljust(FRAME_BYTES, b'\x00')

            if vad.is_speech(r16, VAD_RATE):
                buf.append(raw); speech += 1; silence = 0; active = True
            elif active:
                buf.append(raw); silence += 1
                if silence > 500 // FRAME_MS:
                    if speech > 5:
                        chunk = b"".join(buf)
                        threading.Thread(target=on_speech, args=(chunk,), daemon=True).start()
                    buf, speech, silence, active = [], 0, 0, False
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        stream.stop_stream(); stream.close(); pa.terminate()

main()
