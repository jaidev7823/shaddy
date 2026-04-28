import threading

import numpy as np
import pyaudio
import torch
from silero_vad import load_silero_vad

from config import MIC_RATE, FRAME_MS, VAD_RATE
from worker import worker, get_queue


def main():
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    model = load_silero_vad()
    model.eval()
    
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=1,
        rate=MIC_RATE, input=True,
        frames_per_buffer=int(MIC_RATE * FRAME_MS / 1000)
    )
    print("Listening... (Ctrl+C to stop)")

    buf, speech, silence, active = [], 0, 0, False
    audio_queue = get_queue()
    try:
        while True:
            raw = stream.read(int(MIC_RATE * FRAME_MS / 1000), exception_on_overflow=False)
            arr = np.frombuffer(raw, dtype=np.int16)
            audio_tensor = torch.from_numpy(arr.astype(np.float32) / 32768.0)

            with torch.no_grad():
                speech_prob = model(audio_tensor, VAD_RATE).item()

            if speech_prob > 0.5:
                buf.append(raw); speech += 1; silence = 0; active = True
            elif active:
                buf.append(raw); silence += 1
                if silence > 1500 // FRAME_MS:
                    if speech > 12:
                        chunk = b"".join(buf)
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


if __name__ == "__main__":
    main()
