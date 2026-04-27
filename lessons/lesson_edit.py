#!/usr/bin/env python3
import json, sys
from pathlib import Path

FILE = Path(__file__).with_name("lessons/lessons.json")

def load():
    return json.loads(FILE.read_text()) if FILE.exists() else []

def save(data):
    FILE.write_text(json.dumps(data, indent=2))

def list_():
    for l in load():
        print(f"{l['id']}: {l['topic']} → {l['nudge']}")

def add():
    topic = input("Lesson: ").strip()
    nudge = input("Nudge: ").strip()

    keywords = []
    while (k := input("Keyword (blank to stop): ").strip()):
        keywords.append(k)

    if not (topic and nudge and keywords):
        return

    lid = topic.lower().replace(" ", "_")[:30]

    data = load()
    data.append({
        "id": lid,
        "topic": topic,
        "keywords": keywords,
        "nudge": nudge
    })
    save(data)

def remove(lid):
    save([l for l in load() if l["id"] != lid])

if __name__ == "__main__":
    cmd = sys.argv[1:] or ["list"]

    if cmd[0] == "list":
        list_()
    elif cmd[0] == "add":
        add()
    elif cmd[0] == "remove" and len(cmd) > 1:
        remove(cmd[1])
