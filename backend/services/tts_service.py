import asyncio
import edge_tts
from pydub import AudioSegment
from io import BytesIO

class TTSService:
    def __init__(self):
        self.voice = "en-IN-NeerjaNeural"
        self.output_path = "/tmp/tts_output.wav"
    
    async def generate_audio(self, text: str) -> bytes:
        """Generate audio and return as WAV bytes"""
        try:
            communicate = edge_tts.Communicate(text, voice=self.voice)
            
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            
            if not audio_bytes:
                print("❌ TTS: No audio generated from edge-tts")
                return b""
            
            # Convert MP3 to WAV for better browser compatibility
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
            # Export as WAV bytes
            wav_buffer = BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_bytes = wav_buffer.getvalue()
            
            return wav_bytes
        except Exception as e:
            print(f"❌ TTS Generation Error: {e}")
            return b""
    
    async def speak(self, text: str) -> bool:
        """Generate audio and save to output_path for compatibility"""
        try:
            audio_bytes = await self.generate_audio(text)
            if not audio_bytes:
                return False
            
            # Save to output_path for the /audio/generated endpoint
            with open(self.output_path, "wb") as f:
                f.write(audio_bytes)
            
            return True
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return False