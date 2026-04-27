import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

from prompt import build_prompt

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)


class ESLResponse(BaseModel):
    lesson_id: str
    answer: str
    hint: str


def ask_llm(transcript: str, speaker: str = "student") -> dict:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=build_prompt(transcript, speaker),
            config={
                "system_instruction": "You are a silent ESL coach. Output JSON only.",
                "response_mime_type": "application/json",
                "response_schema": ESLResponse,
                "temperature": 0.1,
            }
        )

        parsed = response.parsed
        nudge = parsed.answer
        should_nudge = bool(nudge and nudge.upper() != "NONE" and len(nudge) > 2)

        return {
            "should_nudge": should_nudge,
            "lesson_id": parsed.lesson_id,
            "nudge": parsed.answer,
            "hint": parsed.hint
        }
    except Exception as e:
        print(f"  Gemini Error: {e}")
        return {"should_nudge": False, "lesson_id": None, "nudge": None}
