#!/usr/bin/env python3
import time, datetime, json, subprocess
import numpy as np
import pyaudio, webrtcvad
from pathlib import Path

# ── Config ─────────────────────────────────────────────
MIC_SAMPLE_RATE = 16000
VAD_RATE        = 16000
FRAME_MS        = 30
FRAME_BYTES     = int(VAD_RATE * FRAME_MS / 1000) * 2

MIN_SPEECH_FRAMES = 10        # ~240ms
MIN_SILENCE_FRAMES = 6       # ~180ms
ENERGY_THRESHOLD = 400       # tune based on your mic
COOLDOWN = 3.0               # seconds

PAUSE_TARGET_MIN = 1.0
PAUSE_TARGET_MAX = 3.0
# ───────────────────────────────────────────────────────

BASE = Path(__file__).parent
LOG  = BASE / "session.log"

def play_beep():
    subprocess.Popen(
        ["aplay", "nudges/beep.wav"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def log_event(delay):
    if delay < PAUSE_TARGET_MIN:
        label = "too_fast"
    elif delay <= PAUSE_TARGET_MAX:
        label = "good"
    else:
        label = "too_slow"

    entry = {
        "t": datetime.datetime.now().isoformat(),
        "delay": round(delay, 2),
        "label": label
    }

    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"→ Delay: {delay:.2f}s ({label})")

def main():
    vad = webrtcvad.Vad(1)
    pa  = pyaudio.PyAudio()

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=MIC_SAMPLE_RATE,
        input=True,
        frames_per_buffer=int(MIC_SAMPLE_RATE * FRAME_MS / 1000)
    )

    print("Listening... Ctrl+C to stop")

    speech_frames = 0
    silence_frames = 0
    in_speech = False

    last_speech_end_time = None
    last_cue_time = 0

    try:
        while True:
            raw = stream.read(int(MIC_SAMPLE_RATE * FRAME_MS / 1000),
                              exception_on_overflow=False)

            arr = np.frombuffer(raw, dtype=np.int16)

            # ── Energy filter ──
            energy = np.abs(arr).mean()
            if energy < ENERGY_THRESHOLD:
                is_speech = False
            else:
                r16 = arr.tobytes()[:FRAME_BYTES].ljust(FRAME_BYTES, b'\x00')
                is_speech = vad.is_speech(r16, VAD_RATE)

            now = time.time()

            # ── Speech tracking ──
            if is_speech:
                speech_frames += 1
                silence_frames = 0
            else:
                silence_frames += 1

            # ── Enter speech state ──
            if not in_speech and speech_frames >= MIN_SPEECH_FRAMES:
                in_speech = True

                # measure response delay
                if last_speech_end_time is not None:
                    delay = now - last_speech_end_time
                    log_event(delay)
                    last_speech_end_time = None

            # ── Exit speech state ──
            if in_speech and silence_frames >= MIN_SILENCE_FRAMES:
                in_speech = False
                speech_frames = 0
                silence_frames = 0

                # mark end of speech
                last_speech_end_time = now

                # trigger cue (cooldown protected)
                if now - last_cue_time > COOLDOWN:
                    play_beep()
                    last_cue_time = now
                    print("• Cue")

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

if __name__ == "__main__":
    main()
