import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

BASE = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIC_RATE = 16000
VAD_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = int(VAD_RATE * FRAME_MS / 1000) * 2
DOWNSAMPLE = MIC_RATE // VAD_RATE

lessons = json.loads((BASE / "lessons/lessons.json").read_text())

COOLDOWN = 10
