import threading

import numpy as np
import pyaudio
import webrtcvad

from config import MIC_RATE, FRAME_MS, DOWNSAMPLE, FRAME_BYTES, VAD_RATE
from worker import worker, get_queue


def main():
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    vad = webrtcvad.Vad(3)
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
            arr16 = arr[::DOWNSAMPLE].astype(np.int16)
            r16 = arr16.tobytes()[:FRAME_BYTES].ljust(FRAME_BYTES, b'\x00')

            if vad.is_speech(r16, VAD_RATE):
                buf.append(raw); speech += 1; silence = 0; active = True
            elif active:
                buf.append(raw); silence += 1
                if silence > 300 // FRAME_MS:
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