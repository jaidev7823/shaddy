import numpy as np
import pyaudio
import torch
from silero_vad import load_silero_vad

from config import MIC_RATE, FRAME_MS, VAD_RATE
print(FRAME_MS)

def main():
    model = load_silero_vad()
    model.eval()

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=1,
        rate=MIC_RATE, input=True,
        frames_per_buffer=int(MIC_RATE * FRAME_MS / 1000)
    )
    print("Listening... (Ctrl+C to stop)")
    print("Speak and watch the speech probability!\n")

    buf, speech, silence, active = [], 0, 0, False
    try:
        while True:
            raw = stream.read(int(MIC_RATE * FRAME_MS / 1000), exception_on_overflow=False)
            arr = np.frombuffer(raw, dtype=np.int16)
            audio_tensor = torch.from_numpy(arr.astype(np.float32) / 32768.0)

            with torch.no_grad():
                speech_prob = model(audio_tensor, VAD_RATE).item()

            if speech_prob > 0.5:
                buf.append(raw); speech += 1; silence = 0; active = True
                print(f"  Speech: {speech_prob:.3f} | Speaking... (frames: {speech})", end="\r")
            elif active:
                buf.append(raw); silence += 1
                print(f"  Speech: {speech_prob:.3f} | Silence... (frames: {silence})", end="\r")
                if silence > 1000 // FRAME_MS:
                    if speech > 12:
                        duration = len(buf) * FRAME_MS / 1000
                        print(f"\n\n✓ Speech detected! Duration: {duration:.2f}s, Frames: {speech}")
                        print(f"  (User stopped speaking after {silence * FRAME_MS}ms of silence)\n")
                    buf, speech, silence, active = [], 0, 0, False
            else:
                if speech_prob > 0.1:
                    print(f"  Speech: {speech_prob:.3f} | Waiting...", end="\r")

    except KeyboardInterrupt:
        print("\n\nDone.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
