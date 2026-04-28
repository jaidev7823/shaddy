import json
from config import PROVIDER, OLLAMA_MODEL, GEMINI_MODEL, GEMINI_API_KEY, get_lessons
from prompt import build_prompt


def _result(answer: str, lesson_id: str, hint: str) -> dict:
    return {
        "should_nudge": bool(answer and answer.upper() != "NONE" and len(answer) > 2),
        "lesson_id": lesson_id,
        "nudge": answer,
        "hint": hint
    }


_EMPTY = {"should_nudge": False, "lesson_id": None, "nudge": None, "hint": None}


def _parse_ollama(response: str) -> dict:
    raw = response.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())
    return _result(data.get("answer", ""), data.get("lesson_id"), data.get("hint"))


def ask_ollama(transcript: str) -> dict:
    import ollama
    r = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "Output JSON only. No markdown."},
            {"role": "user", "content": build_prompt(transcript)}
        ],
        options={
            "num_predict": 60,     # limit tokens
            "temperature": 0.2
        }
    )
    return _parse_ollama(r["message"]["content"])


def ask_gemini(transcript: str) -> dict:
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    r = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_prompt(transcript),
        config={"response_mime_type": "application/json"}
    )
    p = r.parsed
    return _result(p.answer, p.lesson_id, p.hint)


def ask_llm(transcript: str) -> dict:
    try:
        return ask_ollama(transcript) if PROVIDER == "ollama" else ask_gemini(transcript)
    except Exception as e:
        print(f"  LLM Error: {e}")
        return _EMPTY
