"""VAD and speaker verification services."""

import numpy as np
import torch
from silero_vad import load_silero_vad

from backend.config import VAD_RATE, MIC_RATE
from backend.speaker_id import get_embedding, get_student_embedding, _threshold


class VADService:
    """Voice Activity Detection service."""

    def __init__(self):
        self.model = load_silero_vad()
        self.model.eval()

    def detect_speech(self, audio_tensor: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Detect speech probability in audio.
        
        Args:
            audio_tensor: Audio tensor (float32, normalized to [-1, 1])
            threshold: Probability threshold for speech detection
            
        Returns:
            Speech probability (0.0 - 1.0)
        """
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, VAD_RATE).item()
        return speech_prob

    def is_speech(self, audio_tensor: torch.Tensor, threshold: float = 0.5) -> bool:
        """Check if audio contains speech."""
        return self.detect_speech(audio_tensor, threshold) > threshold


class SpeakerVerificationService:
    """Speaker verification service."""

    def __init__(self):
        self.student_embedding = get_student_embedding()
        self.threshold = _threshold

    def is_student_voice(
        self, 
        audio_bytes: bytes, 
        threshold: float = None
    ) -> bool:
        """
        Verify if audio is from the student.
        
        Args:
            audio_bytes: Raw audio bytes (int16)
            threshold: Optional custom threshold
            
        Returns:
            True if speaker is student, False otherwise
        """
        if self.student_embedding is None:
            return False

        try:
            y = np.frombuffer(audio_bytes, dtype=np.int16)
            embedding = get_embedding(y)

            # Cosine similarity
            similarity = np.dot(embedding, self.student_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(self.student_embedding) + 1e-6
            )

            thresh = threshold or self.threshold
            return similarity > thresh
        except Exception as e:
            print(f"Speaker verification error: {e}")
            return False

    def get_speaker_similarity(self, audio_bytes: bytes) -> float:
        """Get speaker similarity score."""
        if self.student_embedding is None:
            return 0.0

        try:
            y = np.frombuffer(audio_bytes, dtype=np.int16)
            embedding = get_embedding(y)

            similarity = np.dot(embedding, self.student_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(self.student_embedding) + 1e-6
            )
            return float(similarity)
        except Exception:
            return 0.0
