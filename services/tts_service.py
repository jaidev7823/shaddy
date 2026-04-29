"""Text-to-Speech service."""

import subprocess
import io

import soundfile as sf
import torch

from config import BASE, DEVICE
from models import tts


class TTSService:
    """Text-to-Speech service."""

    def __init__(self):
        self.tts = tts
        self.output_path = BASE / "lessons/generated.wav"

    def generate_audio(self, text: str) -> bytes:
        """
        Generate audio from text.

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as bytes (WAV format)
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                wav = self.tts.generate(text)

            audio_data = wav.squeeze().cpu().numpy()
            
            # Write to bytes buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, self.tts.sr, format="WAV")
            
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            return buffer.getvalue()

        except Exception as e:
            print(f"TTS error: {e}")
            return b""

    def speak(self, text: str) -> bool:
        """
        Generate audio and play it.

        Args:
            text: Text to speak

        Returns:
            True if successful, False otherwise
        """
        try:
            audio_bytes = self.generate_audio(text)
            if not audio_bytes:
                return False

            # Save and play
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(self.output_path), "wb") as f:
                f.write(audio_bytes)

            subprocess.run(["aplay", "-q", str(self.output_path)], check=True)
            return True

        except Exception as e:
            print(f"TTS error: {e}")
            return False
