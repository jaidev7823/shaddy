import subprocess

import soundfile as sf
import torch

from config import BASE, DEVICE
from models import tts

OUT_PATH = BASE / "lessons/generated.wav"

def speak(text: str):
    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            wav = tts.generate(text)
        
        audio_data = wav.squeeze().cpu().numpy()
        sf.write(str(OUT_PATH), audio_data, tts.sr)
        
        subprocess.run(["aplay", "-q", str(OUT_PATH)], check=True)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  TTS error: {e}")
