import json
from backend.config import PROVIDER, OLLAMA_MODEL, GEMINI_MODEL, GEMINI_API_KEY, get_lessons
from backend.prompt import build_prompt
import ollama
import re

def _result(answer: str, lesson_id: str, why: str, sentence: str) -> dict:
    return {
        "should_nudge": bool(answer and answer.upper() not in ("NONE", "NULL") and len(answer) > 2),
        "lesson_id": lesson_id,
        "nudge": answer,
        "why": why,
        "sentence": sentence
    }

_EMPTY = {"should_nudge": False, "lesson_id": None, "nudge": None, "why": None, "sentence": None}

def _parse_ollama(response: str) -> dict:
    print(f"DEBUG: Raw response: '{response}'")

    if not response or not response.strip():
        return _EMPTY

    raw = response.strip()

    # Strip markdown code blocks (```json ... ``` or ``` ... ```)
    raw = re.sub(r'^```(?:json)?\s*\n?', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\n?```\s*$', '', raw)
    raw = raw.strip()

    # Extract JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON from LLM: {raw}")
        print(f"JSON Error: {e}")
        return _EMPTY

    answer = data.get("answer")
    lesson_id = data.get("lesson_id")
    why = data.get("why")
    sentence = data.get("sentence")

    # Handle null/None values from LLM
    if answer is None or (isinstance(answer, str) and answer.upper() in ("NULL", "NONE")):
        return _EMPTY

    if not isinstance(answer, str):
        answer = str(answer)

    return _result(answer, lesson_id, why, sentence)

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
        data.get("why"),
        data.get("sentence")
    )

def ask_llm(transcript: str) -> dict:
    try:
        return ask_ollama(transcript) if PROVIDER == "ollama" else ask_gemini(transcript)
    except Exception as e:
        print(f"  LLM Error: {e}")
        return _EMPTY
