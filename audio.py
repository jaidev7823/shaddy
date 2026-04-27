import subprocess
from pathlib import Path

import soundfile as sf
import torch

from models import tts_model, VOICE_REF

BASE = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_and_play(text: str):
    try:
        with torch.no_grad():
            wav = tts_model.generate(text, audio_prompt_path=VOICE_REF)
        wav = wav.detach().cpu().contiguous()
        out_path = BASE / "lessons/generated.wav"
        sf.write(str(out_path), wav.squeeze().numpy(), tts_model.sr)
        subprocess.run(["aplay", "-q", str(out_path)])
        del wav
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  TTS error: {e}")
