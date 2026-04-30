"""Transcription service using Faster Whisper."""

import os
import tempfile
import wave

from backend.config import MIC_RATE
from backend.models import whisper


class TranscriptionService:
    """Transcription service using Faster Whisper."""

    def __init__(self):
        self.whisper = whisper

    def transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        language: str = None,
        task: str = "translate",
        vad_filter: bool = True,
    ) -> str:
        """
        Transcribe audio from raw bytes.

        Args:
            audio_bytes: Raw audio data (WAV format in bytes)
            language: Language code (None = auto-detect)
            task: "transcribe" or "translate"
            vad_filter: Enable VAD filtering

        Returns:
            Transcribed text
        """
        print("transcription has been stareted")
        tmp = None
        try:
            # Write bytes to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(MIC_RATE)
                    w.writeframes(audio_bytes)
                tmp = f.name

            # Transcribe
            segs, _ = self.whisper.transcribe(
                tmp,
                language=language,
                task=task,
                vad_filter=vad_filter,
                vad_parameters=dict(min_silence_duration_ms=500),
                temperature=0.0,
                beam_size=5,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
            )

            text = " ".join(s.text for s in segs).strip()
            print(text)
            return text

        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    def transcribe_from_file(self, file_path: str) -> str:
        """Transcribe audio from file."""
        try:
            segs, _ = self.whisper.transcribe(
                file_path,
                language=None,
                task="translate",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                temperature=0.0,
                beam_size=5,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
            )
            return " ".join(s.text for s in segs).strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
