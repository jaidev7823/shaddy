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
        communicate = edge_tts.Communicate(text, voice=self.voice)

        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        # Load MP3 from memory
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")

        # Convert to raw PCM for playback
        samples = np.array(audio.get_array_of_samples())

        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
        
        return True
