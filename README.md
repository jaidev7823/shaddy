# Coach - ESL Shadow Assistant

Real-time speech coaching for ESL students using Whisper, Gemini, and Chatterbox TTS.

## Modules

| File | Responsibility |
|-----|---------------|
| `config.py` | Config, constants, env vars |
| `models.py` | TTS & Whisper model loading |
| `prompt.py` | Prompt builder for LLM |
| `llm.py` | Gemini LLM integration |
| `audio.py` | TTS generation & playback |
| `worker.py` | Queue worker, transcription loop |
| `main.py` | Main mic loop entry point |

## Usage

```bash
python main.py
```

## Requirements

Set `GEMINI_API_KEY` in `.env` file.