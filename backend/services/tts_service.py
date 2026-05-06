import asyncio
import edge_tts
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
from io import BytesIO

class TTSService:
    def __init__(self):
        self.voice = "en-IN-NeerjaNeural"
    
    async def speak(self, text: str):
        try:
            communicate = edge_tts.Communicate(text, voice=self.voice)
            
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            
            if not audio_bytes:
                print("❌ TTS: No audio generated from edge-tts")
                return False
            
            # Load MP3 from memory
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
            
            # Convert to raw PCM for playback
            samples = np.array(audio.get_array_of_samples())
            
            sd.play(samples, samplerate=audio.frame_rate)
            await asyncio.to_thread(sd.wait)
            
            return True
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return False
