from chatterbox.tts_turbo import ChatterboxTurboTTS
from faster_whisper import WhisperModel

from backend.config import DEVICE

whisper = WhisperModel(
    "models/faster-whisper-large-v3",
    device="cuda",
    compute_type="float16"
)
tts = ChatterboxTurboTTS.from_pretrained(device=DEVICE)

