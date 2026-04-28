import subprocess

import soundfile as sf
import torch

from config import BASE, DEVICE
from models import tts

OUT_PATH = BASE / "lessons/generated.wav"


def speak(text: str):
    try:
        with torch.no_grad():
            wav = tts.generate(text)
        sf.write(str(OUT_PATH), wav.squeeze().cpu().numpy(), tts.sr)
        subprocess.run(["aplay", "-q", str(OUT_PATH)])
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  TTS error: {e}")