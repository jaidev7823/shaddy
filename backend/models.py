from chatterbox.tts_turbo import ChatterboxTurboTTS
from faster_whisper import WhisperModel

from backend.config import DEVICE

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models" / "faster-whisper-large-v3"

whisper = WhisperModel(
    str(MODEL_DIR),
    device="cuda",
    compute_type="float16"
)

# Use medium model directly (downloads automatically if not cached)
# whisper = WhisperModel(
#     "medium",
#     device="cuda",
#     compute_type="float16"
# )

tts = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
