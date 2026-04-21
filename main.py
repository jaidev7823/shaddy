#!/usr/bin/env python3
import json, subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

BASE = Path(__file__).parent
LOG = BASE / "session.log"
LESSONS = BASE / "lessons.json"

def today_events():
    if not LOG.exists():
        exit("No session log found.")

    today = datetime.now().date()
    out = []
    for line in LOG.read_text().splitlines():
        try:
            e = json.loads(line)
            if datetime.fromisoformat(e["time"]).date() == today:
                out.append(e)
        except:
            pass
    return out

def summarize(events):
    t = [e["text"] for e in events if e["event"] == "transcript"]
    n = [e for e in events if e["event"] == "nudge_triggered"]
    return {
        "sessions": sum(e["event"] == "session_start" for e in events),
        "utterances": len(t),
        "nudges": len(n),
        "breakdown": Counter(x["lesson_id"] for x in n),
        "moments": [{"lesson": x["lesson_id"], "when": x["transcript"]} for x in n][:5],
        "samples": t[:8],
    }

def ollama(prompt, model="mistral"):
    try:
        r = subprocess.run(["ollama", "run", model],
                           input=prompt, text=True,
                           capture_output=True, timeout=120)
        return r.stdout.strip()
    except Exception as e:
        return f"[ollama error: {e}]"

def main():
    ev = today_events()
    if not ev:
        return print("No events today.")

    s = summarize(ev)
    lessons = {l["id"]: l["topic"] for l in json.loads(LESSONS.read_text())}

    print(f"\nSessions: {s['sessions']}")
    print(f"Utterances: {s['utterances']}")
    print(f"Nudges: {s['nudges']}")

    if s["breakdown"]:
        print("\nBreakdown:")
        for k,v in s["breakdown"].items():
            print(f"  {lessons.get(k,k)}: {v}")

    if s["moments"]:
        print("\nMoments:")
        for m in s["moments"]:
            print(f"  [{lessons.get(m['lesson'], m['lesson'])}] {m['when'][:80]}")

    prompt = f"""Review this session briefly.

Stats: {s['utterances']} utterances, {s['nudges']} nudges
Lessons: {list(lessons.values())}
Moments: {json.dumps(s['moments'], indent=2)}
Samples: {json.dumps(s['samples'], indent=2)}

Give:
- patterns
- one strength
- one focus
- 3-item checklist"""

    print("\n--- AI Review ---\n")
    print(ollama(prompt))
    print(f"\nLog: {LOG}")

if __name__ == "__main__":
    main()
