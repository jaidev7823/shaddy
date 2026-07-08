from backend.config import get_lessons

def build_prompt(transcript: str) -> str:
    lessons = get_lessons()
    words = [l["topic"] for l in lessons]
    words_str = ", ".join(words)
    
    return f"""You are a silent vocabulary spotter. Your default answer is null.

Today's words: {words_str}

Someone just said: "{transcript}"

ONLY return a word if this is GENUINELY a natural moment to use it.
Most inputs should return null. Silence is better than a forced suggestion.

Ask yourself: "Would a native English speaker actually use this word 
in a reply to what was just said?" If any doubt — return null.

GOOD moment to trigger:
Input: "What did you think of the movie?"
Word: entertainment
Trigger: YES — movie is literally entertainment

BAD moment — return null:
Input: "This is how it works"
Word: pragmatic  
Trigger: NO — pragmatic doesn't naturally fit here

Input: "Can you pass the water?"
Word: resilient
Trigger: NO — nothing connects

### OUTPUT — valid JSON only, no explanation:
{{"lesson_id": "string or null", "answer": "string or null", "sentence": "string or null", "why": "string or null"}}"""
