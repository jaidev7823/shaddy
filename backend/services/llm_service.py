"""LLM integration service."""

from backend.llm import ask_llm, ask_ollama, ask_gemini


class LLMService:
    """LLM service for processing transcriptions."""

    def process_transcript(self, transcript: str) -> dict:
        """
        Process transcript through LLM.

        Args:
            transcript: Transcribed text

        Returns:
            Dict with keys:
            - should_nudge: bool
            - lesson_id: str or None
            - nudge: str or None
            - why: str or None
        """
        return ask_llm(transcript)

    def process_with_ollama(self, transcript: str) -> dict:
        """Process using Ollama."""
        return ask_ollama(transcript)

    def process_with_gemini(self, transcript: str) -> dict:
        """Process using Gemini."""
        print("starting gemini")
        return ask_gemini(transcript)
