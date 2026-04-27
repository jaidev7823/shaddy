from pathlib import Path

import torch
from faster_whisper import WhisperModel
from chatterbox.tts_turbo import ChatterboxTurboTTS

BASE = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tts_model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
whisper = WhisperModel("medium", device="cuda", compute_type="float16")
VOICE_REF = str(BASE / "audio/l_voice_sample.wav")
