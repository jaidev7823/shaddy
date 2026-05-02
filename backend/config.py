import os
import json
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIC_RATE = 16000
VAD_RATE = 16000
FRAME_MS = 32
FRAME_BYTES = int(VAD_RATE * FRAME_MS / 1000) * 2
DOWNSAMPLE = MIC_RATE // VAD_RATE
CHUNK_SIZE = FRAME_BYTES * DOWNSAMPLE

COOLDOWN = 10

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:latest")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

VOICE_REF = str(BASE / "audio/l_voice_sample.wav")
SAMPLE_VOICE_STUDENT = str(BASE / "audio/my_voice_sample.wav") 
LESSONS_PATH = BASE / "lessons/lessons.json"
TRIGGER_LIMIT = 800 // FRAME_MS

def get_lessons():
    return json.loads(LESSONS_PATH.read_text())
