# AI Learning Coach

Listens to your conversations, whispers lesson reminders in your ear when the moment is right.

## How it works

```
You read something in the morning → Add it as a lesson with a keyword
↓
During conversation someone says a trigger phrase
↓
Whisper tiny.en transcribes it in ~200ms
↓
Keyword match finds relevant lesson
↓
Pre-baked audio plays in your earbuds immediately
↓
Session logged → evening review shows what happened
```

## Setup (one time)

```bash
# 1. Install system deps
sudo apt install espeak-ng portaudio19-dev python3-pip

# 2. Install Python deps
pip3 install faster-whisper webrtcvad pyaudio

# 3. Whisper model downloads automatically on first run (~40MB for tiny.en)
```

## Daily flow

### Morning — add today's lesson

```bash
python3 lessons_edit.py add
```

You'll be prompted for:
- What the lesson is
- What the whisper should say
- What keywords trigger it (things OTHER people say to you)

### During the day — run the coach

```bash
python3 coach.py
```

- It lists your audio devices — pick your mic
- Run it in a terminal in the background while you go about your day
- When someone says a trigger phrase, you'll hear the whisper in your earbuds

**Bluetooth tip:** If your earbuds show up as two devices (one for audio, one for mic/headset), use the headset profile as your **output device** for lower latency. Set it in your system audio settings.

### Evening — review your session

```bash
python3 review.py
```

Shows:
- How many times each lesson was triggered
- What conversations triggered it
- AI critique via ollama (needs `ollama run mistral` installed)
- Checklist for tomorrow

## Latency breakdown

```
VAD detects end of utterance   ~0ms
Whisper tiny.en transcribe     ~150-300ms
Keyword match                  ~5ms
Audio playback (pre-baked)     ~50ms
────────────────────────────────────
Total from end of sentence     ~200-350ms
```

This is fast enough. By the time they finish the question you have the whisper.

## Tuning

Edit `coach.py` top section:

```python
VAD_AGGRESSIVENESS = 2      # 0=gentle, 3=aggressive (higher = less false triggers)
SPEECH_BUFFER_SECONDS = 3   # how much audio to transcribe at once
COOLDOWN_SECONDS = 30       # min seconds between same lesson nudge
MIN_SPEECH_FRAMES = 10      # how many frames = "real speech" (avoid noise triggers)
```

## File structure

```
coach/
├── coach.py          # main listener
├── review.py         # post-session critique
├── lessons_edit.py   # add/remove lessons
├── lessons.json      # your lessons (edit this)
├── nudges/           # pre-generated audio files (auto-created)
│   └── *.wav
└── session.log       # everything that happened (JSON lines)
```

## Lessons format

```json
{
  "id": "think_before_speak",
  "topic": "Think 3 seconds before speaking",
  "keywords": ["what do you think", "your opinion", "what would you"],
  "nudge": "pause. think first. then speak."
}
```

- `keywords` — phrases the OTHER person says that mean you should apply the lesson
- `nudge` — exactly what gets whispered in your ear (keep it short — 3-6 words)
- Add as many lessons as you want, they all run simultaneously

