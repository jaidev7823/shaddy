import asyncio
import subprocess
import edge_tts
from backend.config import BASE

class TTSService:
    def __init__(self):
        self.output_path = BASE / "lessons/generated.wav"
        self.voice = "en-IN-NeerjaNeural"  # Indian English, sounds natural

    def speak(self, text: str) -> bool:
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            asyncio.run(self._generate(text))
            subprocess.run(["aplay", "-q", str(self.output_path)], check=True)
            return True
        except Exception as e:
            print(f"TTS error: {e}")
            return False

    async def _generate(self, text: str):
        communicate = edge_tts.Communicate(text, voice=self.voice)
        await communicate.save(str(self.output_path))
