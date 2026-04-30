"""VAD and Speaker Verification Services using SpeechBrain."""

import numpy as np
import torch
import os

from silero_vad import load_silero_vad

# Import only what's needed from the new speaker_id module
from backend.speaker_id import (
    get_verification_model,
    get_student_embedding,
    _threshold as DEFAULT_SPEAKER_THRESHOLD
)


class VADService:
    """Voice Activity Detection service using Silero VAD."""

    def __init__(self):
        self.model = load_silero_vad()
        self.model.eval()
        print("✅ Silero VAD model loaded successfully.")

    def detect_speech(self, audio_tensor: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Detect speech probability in a given audio chunk.
        
        Args:
            audio_tensor: Torch tensor of shape [samples] or [1, samples], 16kHz
            threshold: Probability threshold (default 0.5)
            
        Returns:
            Speech probability (float between 0 and 1)
        """
        try:
            print(f"🔍 VAD DEBUG: Input shape={audio_tensor.shape}, dtype={audio_tensor.dtype}")

            # Ensure tensor is on CPU and float32 (Silero VAD requirement)
            audio_tensor = audio_tensor.float().cpu()

            # Silero VAD works best with specific chunk sizes: 512, 1024, or 1536
            target_size = 512
            current_size = audio_tensor.shape[-1]

            if current_size > target_size:
                audio_tensor = audio_tensor[..., :target_size]
                print(f"🔍 VAD DEBUG: Truncated to {target_size} samples")
            elif current_size < target_size:
                padding = target_size - current_size
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
                print(f"🔍 VAD DEBUG: Padded to {target_size} samples")

            # Add batch dimension if missing → [1, 512]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            with torch.no_grad():
                speech_prob = self.model(audio_tensor, 16000).item()

            print(f"🔍 VAD DEBUG: Speech probability = {speech_prob:.4f} | Above threshold? {speech_prob > threshold}")
            return speech_prob

        except Exception as e:
            print(f"❌ VAD Error: {e}")
            return 0.0


class SpeakerVerificationService:
    """Speaker Verification service using SpeechBrain ECAPA-TDNN."""

    def __init__(self):
        self.student_embedding = get_student_embedding()
        self.threshold = DEFAULT_SPEAKER_THRESHOLD
        self.model = None  # Lazy load

        if self.student_embedding is None:
            print("⚠️ Warning: Student voice reference could not be loaded.")
        else:
            print(f"✅ Student voice enrolled | Threshold: {self.threshold:.3f}")

    def _get_model(self):
        """Lazy initialization of SpeechBrain model."""
        if self.model is None:
            self.model = get_verification_model()
        return self.model

    def is_student_voice(
        self,
        audio_bytes: bytes,
        threshold: float = None
    ) -> bool:
        """
        Check if the given audio bytes belong to the enrolled student.
        """
        if self.student_embedding is None:
            print("⚠️ Student embedding not available.")
            return False

        thresh = threshold or self.threshold

        try:
            model = self._get_model()
            test_tensor = self._audio_bytes_to_tensor(audio_bytes)

            # Get embedding for test audio
            test_embedding = model.encode_batch(test_tensor)
            test_embedding = test_embedding.squeeze(0).squeeze(0)

            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                self.student_embedding.unsqueeze(0),
                test_embedding.unsqueeze(0),
                dim=1
            ).item()

            is_match = similarity > thresh

            print(f"🎤 Speaker Verification: similarity = {similarity:.4f} | "
                  f"threshold = {thresh} → {'✅ Match' if is_match else '❌ No match'}")

            return is_match

        except Exception as e:
            print(f"❌ Speaker verification error: {e}")
            return False

    def get_speaker_similarity(self, audio_bytes: bytes) -> float:
        """Return raw cosine similarity score (useful for debugging)."""
        if self.student_embedding is None:
            return 0.0

        try:
            model = self._get_model()
            test_tensor = self._audio_bytes_to_tensor(audio_bytes)

            test_embedding = model.encode_batch(test_tensor)
            test_embedding = test_embedding.squeeze(0).squeeze(0)

            similarity = torch.nn.functional.cosine_similarity(
                self.student_embedding.unsqueeze(0),
                test_embedding.unsqueeze(0),
                dim=1
            ).item()

            print(f"🎯 Speaker similarity: {similarity:.4f} (threshold = {self.threshold})")
            return float(similarity)

        except Exception as e:
            print(f"❌ Similarity computation error: {e}")
            return 0.0

    def _audio_bytes_to_tensor(self, audio_bytes: bytes) -> torch.Tensor:
        """Convert raw int16 PCM bytes to torch tensor [1, time]."""
        # Convert bytes → numpy float32 normalized
        y = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Ensure mono
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        tensor = torch.from_numpy(y).unsqueeze(0)   # [1, time]
        return tensor.to(self._get_model().device)


# Optional: Combined service for convenience
class AudioProcessingService:
    """Combined VAD + Speaker Verification service."""

    def __init__(self):
        self.vad = VADService()
        self.speaker = SpeakerVerificationService()

    def process_chunk(self, audio_bytes: bytes, vad_threshold=0.5, speaker_threshold=None):
        """Process one audio chunk: VAD + Speaker Verification."""
        # Convert bytes to tensor for VAD
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        speech_prob = self.vad.detect_speech(audio_tensor, threshold=vad_threshold)

        if speech_prob > vad_threshold:
            is_student = self.speaker.is_student_voice(audio_bytes, threshold=speaker_threshold)
            similarity = self.speaker.get_speaker_similarity(audio_bytes)
            return {
                "has_speech": True,
                "speech_prob": speech_prob,
                "is_student": is_student,
                "similarity": similarity
            }
        else:
            return {
                "has_speech": False,
                "speech_prob": speech_prob,
                "is_student": False,
                "similarity": 0.0
            }
