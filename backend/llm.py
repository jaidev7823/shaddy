import json
from backend.config import PROVIDER, OLLAMA_MODEL, GEMINI_MODEL, GEMINI_API_KEY, get_lessons
from backend.prompt import build_prompt
import ollama
import re

def _result(answer: str, lesson_id: str, why: str) -> dict:
    return {
        "should_nudge": bool(answer and answer.upper() != "NONE" and len(answer) > 2),
        "lesson_id": lesson_id,
        "nudge": answer,
        "why":  why 
    }

_EMPTY = {"should_nudge": False, "lesson_id": None, "nudge": None, "why": None}



def _parse_ollama(response: str) -> dict:
    print(f"DEBUG: Raw response: '{response}'")

    if not response or not response.strip():
        return _EMPTY

    raw = response.strip()

    # Extract JSON object only
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("Invalid JSON from Ollama:", raw)
        return _EMPTY

    answer = data.get("answer")

    # HARD GUARD: answer must be single word or None
    if not answer or not isinstance(answer, str) or " " in answer:
        answer = None

    return _result(
        answer,
        data.get("lesson_id") if answer else None,
        data.get("why") if answer else None
    )

def ask_ollama(transcript: str) -> dict:
    # Use generate instead of chat to simplify the request structure
    r = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=build_prompt(transcript)
    )
    # Note: 'generate' uses ['response'], 'chat' uses ['message']['content']
    content = r.get("response", "") 
    print(f"DEBUG: Raw response: '{content}'")
    return _parse_ollama(content)

def ask_gemini(transcript: str) -> dict:
    print("using gemini to talk")
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_prompt(transcript),
        config={
            "response_mime_type": "application/json"
        }
    )

    text = response.text.strip()
    print(f"DEBUG: Raw response: '{text}'")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("Invalid JSON from Gemini:", text)
        return _EMPTY

    return _result(
        data.get("answer", ""),
        data.get("lesson_id"),
        data.get("why")
    )

def ask_llm(transcript: str) -> dict:
    try:
        return ask_ollama(transcript) if PROVIDER == "ollama" else ask_gemini(transcript)
    except Exception as e:
        print(f"  LLM Error: {e}")
        return _EMPTY
