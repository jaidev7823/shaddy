"""VAD and Speaker Verification Services using SpeechBrain."""

import numpy as np
import torch
import os

from silero_vad import load_silero_vad
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
        print("✅ Silero VAD model loaded.")

    def detect_speech(self, audio_tensor: torch.Tensor, threshold: float = 0.5) -> float:
        """Detect speech probability in a given audio chunk."""
        try:
            audio_tensor = audio_tensor.float().cpu()

            # Silero VAD requirement: target size 512
            target_size = 512
            current_size = audio_tensor.shape[-1]

            if current_size > target_size:
                audio_tensor = audio_tensor[..., :target_size]
            elif current_size < target_size:
                padding = target_size - current_size
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            with torch.no_grad():
                speech_prob = self.model(audio_tensor, 16000).item()

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
            print("⚠️ Warning: Student voice reference not found.")
        else:
            print(f"✅ Student voice enrolled (Threshold: {self.threshold:.3f})")

    def _get_model(self):
        if self.model is None:
            self.model = get_verification_model()
        return self.model

    def is_student_voice(self, audio_bytes: bytes, threshold: float = None) -> bool:
        """Check if audio belongs to the enrolled student."""
        if self.student_embedding is None:
            return False

        thresh = threshold or self.threshold

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

            is_match = similarity > thresh
            
            # Keep this: Important to know the result of the verification check
            status = "✅ Match" if is_match else "❌ No match"
            print(f"🎤 Speaker Verification: {status} (Score: {similarity:.4f})")

            return is_match

        except Exception as e:
            print(f"❌ Speaker verification error: {e}")
            return False

    def get_speaker_similarity(self, audio_bytes: bytes) -> float:
        """Return raw cosine similarity score."""
        if self.student_embedding is None:
            return 0.0

        try:
            model = self._get_model()
            test_tensor = self._audio_bytes_to_tensor(audio_bytes)
            test_embedding = model.encode_batch(test_tensor).squeeze(0).squeeze(0)

            similarity = torch.nn.functional.cosine_similarity(
                self.student_embedding.unsqueeze(0),
                test_embedding.unsqueeze(0),
                dim=1
            ).item()

            return float(similarity)
        except Exception:
            return 0.0

    def _audio_bytes_to_tensor(self, audio_bytes: bytes) -> torch.Tensor:
        y = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        return torch.from_numpy(y).unsqueeze(0).to(self._get_model().device)


class AudioProcessingService:
    """Combined VAD + Speaker Verification service."""

    def __init__(self):
        self.vad = VADService()
        self.speaker = SpeakerVerificationService()

    def process_chunk(self, audio_bytes: bytes, vad_threshold=0.5, speaker_threshold=None):
        """Process one audio chunk: VAD + Speaker Verification."""
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        speech_prob = self.vad.detect_speech(audio_tensor, threshold=vad_threshold)

        if speech_prob > vad_threshold:
            # We only perform speaker ID and print if speech is actually detected
            similarity = self.speaker.get_speaker_similarity(audio_bytes)
            is_student = similarity > (speaker_threshold or self.speaker.threshold)
            
            return {
                "has_speech": True,
                "speech_prob": round(speech_prob, 4),
                "is_student": is_student,
                "similarity": round(similarity, 4)
            }
        
        return {
            "has_speech": False,
            "speech_prob": round(speech_prob, 4),
            "is_student": False,
            "similarity": 0.0
        }
