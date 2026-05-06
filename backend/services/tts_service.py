import wave
from io import BytesIO
from piper import PiperVoice
import numpy as np

class TTSService:
    def __init__(self):
        self.model_path = "backend/models/en_US-lessac-medium.onnx"
        self.voice = PiperVoice.load(self.model_path)

        self.output_path = "/tmp/tts_output.wav"

    async def stream_audio(self, text: str):
        """Yields raw PCM audio chunks (Int16) as they are generated."""
        try:
            # Piper's synthesize() returns a generator of AudioChunk objects
            # Each AudioChunk has audio_int16_bytes attribute with raw PCM data
            # We yield smaller chunks for better streaming performance
            chunk_size = 4096  # bytes per chunk (2048 Int16 samples)
            
            for audio_chunk in self.voice.synthesize(text):
                if not audio_chunk or not hasattr(audio_chunk, 'audio_int16_bytes'):
                    continue
                
                pcm_bytes = audio_chunk.audio_int16_bytes
                if not pcm_bytes:
                    continue
                
                # Split large chunks into smaller ones for streaming
                for i in range(0, len(pcm_bytes), chunk_size):
                    yield pcm_bytes[i:i+chunk_size]
        except Exception as e:
            print(f"❌ TTS Stream Error: {e}")
            import traceback
            traceback.print_exc()

    async def generate_audio(self, text: str) -> bytes:
        """Generate WAV audio bytes using Piper"""

        try:
            wav_buffer = BytesIO()

            # Piper writes directly into WAV file object
            with wave.open(wav_buffer, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)

            wav_bytes = wav_buffer.getvalue()

            if not wav_bytes:
                print("❌ Piper: No audio generated")
                return b""

            return wav_bytes

        except Exception as e:
            print(f"❌ Piper TTS Generation Error: {e}")
            return b""

    async def speak(self, text: str) -> bytes:
        """Generate audio and return bytes directly"""
        try:
            audio_bytes = await self.generate_audio(text)
            if not audio_bytes:
                return b""
            return audio_bytes
        except Exception as e:
            print(f"❌ Piper TTS Error: {e}")
            return b""
