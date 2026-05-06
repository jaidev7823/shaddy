from backend.config import get_lessons

def build_prompt(transcript: str) -> str:
    lessons = get_lessons()
    word = "\n".join(f'{l["id"]}: {l["topic"]}' for l in lessons)

    return f"""You are a silent vocabulary spotter.

Today's user wants to learn:
{word}

Someone just said: "{transcript}"

Can the student naturally use one of today's words in their reply?

If YES — return JSON with:
- the word
- a short natural sentence using that word
- a short reason

If NO — return JSON with nulls.

STRICT Rules:
- answer must be ONE word from today's list or null
- sentence must be a short natural reply using that word
- why must be short (few words only)
- If answer is null → sentence and why must also be null
- No explanation outside JSON
- If not 100% sure → return null
- Do not force usage
- The sentence MUST contain the exact answer word (case-insensitive)
- If the word is not present → return null for all fields

BAD:
answer: "Pragmatic"
sentence: "I try to be practical."
→ INVALID (word missing)

GOOD:
answer: "Pragmatic"
sentence: "I'm pragmatic about my friendships."

### OUTPUT — valid JSON only:
{{
  "lesson_id": "string or null",
  "answer": "string or null",
  "sentence": "string or null",
  "why": "string or null"
}}"""
