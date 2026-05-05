import base64
import numpy as np
import torch
import torchaudio
from typing import Optional, Tuple

class AudioProcessor:
    def __init__(self, vad_service):
        self.vad_service = vad_service
        self.resamplers = {}
        self.target_rate = 16000
    
    def process_chunk(self, audio_b64: str, sample_rate: int = 16000) -> Tuple[Optional[dict], Optional[str]]:
        """Process an audio chunk. Returns (result, error_message)"""
        if not audio_b64:
            return None, "No audio data in chunk"
        
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return None, f"Base64 decode failed: {str(e)}"
        
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]
        if not audio_bytes:
            return None, "Empty audio after trimming"
        
        try:
            arr = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_tensor = torch.from_numpy(arr.astype(np.float32) / 32768.0)
            
            # Resample if needed
            if sample_rate != self.target_rate:
                if sample_rate not in self.resamplers:
                    print(f"Creating resampler: {sample_rate} -> {self.target_rate}")
                    self.resamplers[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, 
                        new_freq=self.target_rate
                    )
                
                # Ensure audio_tensor is in correct shape for torchaudio (1, samples)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                audio_tensor = self.resamplers[sample_rate](audio_tensor)
                
                # Update audio_bytes for the rest of the pipeline
                resampled_np = (audio_tensor.squeeze().cpu().numpy() * 32768.0).astype(np.int16)
                audio_bytes = resampled_np.tobytes()
            
        except Exception as e:
            return None, f"Audio conversion/resampling failed: {str(e)}"
        
        try:
            speech_prob = self.vad_service.detect_speech(audio_tensor)
        except Exception as e:
            return None, f"VAD detection failed: {str(e)}"
        
        return {
            "speech_prob": speech_prob,
            "audio_bytes": audio_bytes,
            "audio_tensor": audio_tensor
        }, None
