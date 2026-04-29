# FastAPI Migration Guide

## 📋 What Changed

Your project has been migrated from standalone Python scripts to a modern **FastAPI** web service architecture.

### Before (Script-based)
```
main.py
  ↓ (starts thread)
  ├─ worker.py (background queue processing)
  └─ Blocking mic loop + manual threading
```

### After (FastAPI)
```
app.py (FastAPI server)
  ├─ HTTP endpoints (REST)
  │   ├─ GET /health (health check)
  │   ├─ POST /process-audio (file upload)
  │   └─ GET /audio/generated (get response audio)
  │
  ├─ WebSocket endpoint (/ws/audio)
  │   └─ Real-time audio streaming
  │
  └─ services/ (modular, reusable)
      ├─ vad_service.py (VAD + speaker verification)
      ├─ transcription_service.py (Whisper)
      ├─ llm_service.py (LLM integration)
      └─ tts_service.py (Text-to-speech)
```

---

## 🚀 Getting Started

### 1. Install New Dependencies

```bash
pip install -r requirements.txt
```

New packages added:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `websockets` - WebSocket support
- `python-multipart` - File uploads
- `websockets` - WebSocket protocol

### 2. Run the Server

```bash
# Development (auto-reload on changes)
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Access the API

- **Interactive Docs**: http://localhost:8000/docs
- **Root**: http://localhost:8000/
- **Health Check**: http://localhost:8000/health

---

## 📡 API Endpoints

### REST Endpoints

#### `GET /health`
Health check endpoint.

```bash
curl -X GET http://localhost:8000/health | jq .
```

Response:
```json
{
  "status": "ok",
  "models_loaded": true,
  "student_enrolled": true
}
```

#### `POST /process-audio`
Process an audio file through the full pipeline.

```bash
curl -X POST -F 'file=@audio.wav' http://localhost:8000/process-audio | jq .
```

Response:
```json
{
  "transcript": "Hello, can you help me?",
  "speaker_is_student": false,
  "speaker_similarity": 0.45,
  "llm_response": {
    "should_nudge": true,
    "lesson_id": "lesson_1",
    "nudge": "Remember to try the method we discussed",
    "why": "Student is struggling with the concept"
  },
  "audio_generated": true
}
```

#### `GET /audio/generated`
Get the last generated response audio.

```bash
curl -X GET http://localhost:8000/audio/generated --output response.wav
```

---

### WebSocket Endpoint

#### `WS /ws/audio`
Real-time audio streaming with live responses.

**Python Client:**
```bash
python examples/websocket_client.py audio.wav
```

**Message Format (Client → Server):**
```json
{
  "type": "audio_chunk",
  "data": {
    "audio": "<base64 encoded audio bytes>",
    "sample_rate": 16000
  }
}
```

**Response Format (Server → Client):**
```json
{
  "type": "status|response|error",
  "data": {
    "state": "listening|processing|transcribed",
    "text": "...",
    "speaker_similarity": 0.95,
    "llm_response": {...},
    "cooldown_active": false
  }
}
```

---

## 🏗️ Project Structure

```
shady/
├── app.py                          # Main FastAPI application
├── config.py                       # Configuration (unchanged)
├── models.py                       # Model loading (unchanged)
├── prompt.py                       # LLM prompts (unchanged)
├── llm.py                          # LLM integration (unchanged)
├── audio.py                        # Audio utilities (unchanged)
├── speaker_id.py                   # Speaker verification (unchanged)
│
├── schemas.py                      # Pydantic models (NEW)
├── services/                       # Service layer (NEW)
│   ├── __init__.py
│   ├── vad_service.py              # VAD + speaker verification
│   ├── transcription_service.py    # Whisper transcription
│   ├── llm_service.py              # LLM processing
│   └── tts_service.py              # Text-to-speech
│
├── examples/                       # Example clients (NEW)
│   ├── websocket_client.py         # WebSocket client
│   └── curl_examples.sh            # Curl examples
│
├── requirements.txt                # Updated with FastAPI deps
├── MIGRATION_GUIDE.md              # This file
└── [legacy files]
    ├── main.py                     # Can be archived
    └── worker.py                   # Can be archived
```

---

## 📚 Service Layer

Services are now modular and reusable:

### VADService
```python
from services import VADService

vad = VADService()
speech_prob = vad.detect_speech(audio_tensor)
is_speech = vad.is_speech(audio_tensor, threshold=0.5)
```

### SpeakerVerificationService
```python
from services import SpeakerVerificationService

speaker = SpeakerVerificationService()
is_student = speaker.is_student_voice(audio_bytes)
similarity = speaker.get_speaker_similarity(audio_bytes)
```

### TranscriptionService
```python
from services import TranscriptionService

transcriber = TranscriptionService()
text = transcriber.transcribe_from_bytes(audio_bytes)
text = transcriber.transcribe_from_file("audio.wav")
```

### LLMService
```python
from services import LLMService

llm = LLMService()
result = llm.process_transcript("transcribed text")
```

### TTSService
```python
from services import TTSService

tts = TTSService()
audio_bytes = tts.generate_audio("Hello!")
tts.speak("Hello!")
```

---

## 🔄 Migration Path

### Old Code (Blocked Loop)
```python
# main.py (old)
while True:
    raw = stream.read(...)
    speech_prob = model(audio_tensor).item()
    
    if speech_prob > 0.5:
        audio_queue.put(raw)
```

### New Code (Async)
```python
# app.py (WebSocket)
@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    async for msg in websocket:
        audio_tensor = process(msg)
        speech_prob = vad_service.detect_speech(audio_tensor)
        
        if speech_prob > 0.5:
            await websocket.send_json({...})
```

---

## 💡 Key Benefits

| Feature | Before | After |
|---------|--------|-------|
| **Multiple Clients** | ❌ Single mic only | ✅ Unlimited WebSockets |
| **API Access** | ❌ None | ✅ REST + WebSocket |
| **Real-time Streaming** | ⚠️ Manual threads | ✅ Native async |
| **Error Handling** | ⚠️ Basic | ✅ Comprehensive |
| **Documentation** | ❌ None | ✅ Auto Swagger UI |
| **Scalability** | ⚠️ Limited | ✅ Highly scalable |
| **Testing** | ⚠️ Manual | ✅ Can use pytest |
| **Deployment** | ⚠️ Difficult | ✅ Docker-ready |

---

## 🧪 Testing

### Quick Test
```bash
# 1. Start server
python -m uvicorn app:app --reload

# 2. In another terminal, health check
curl http://localhost:8000/health

# 3. Process a file
curl -X POST -F 'file=@test.wav' http://localhost:8000/process-audio

# 4. WebSocket test
python examples/websocket_client.py test.wav
```

### Interactive Swagger UI
Open: http://localhost:8000/docs

---

## 🐳 Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    alsa-utils \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t shady-api .
docker run -p 8000:8000 --device /dev/snd shady-api
```

---

## 🔧 Configuration

All configuration remains in `config.py`:
```python
MIC_RATE = 16000
VAD_RATE = 16000
FRAME_MS = 32
COOLDOWN = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

Environment variables (`.env`):
```
LLM_PROVIDER=ollama  # or "gemini"
OLLAMA_MODEL=gemma4:latest
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
```

---

## 📝 Legacy Code

The old `main.py` and `worker.py` are still present but not used:
- `main.py` → Replaced by `app.py`
- `worker.py` → Logic integrated into services + API endpoints

You can keep them for reference or delete after migration is complete.

---

## 🚨 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Use different port
python -m uvicorn app:app --port 8001
```

### Models not loading
```bash
# Check models directory exists
ls models/

# Verify model files are intact
ls models/faster-whisper-large-v3/
```

### WebSocket connection refused
```bash
# Ensure server is running
curl http://localhost:8000/health

# Check firewall
sudo ufw allow 8000
```

### Audio files not processing
```bash
# Check file format
file test.wav
# Should output: test.wav: RIFF (little-endian) data, WAVE...

# Try with sample rate 16000
ffmpeg -i input.wav -ar 16000 test.wav
```

---

## 🎯 Next Steps

1. ✅ **Verify Migration**: Run `curl http://localhost:8000/health`
2. ✅ **Test REST**: Upload audio file and verify processing
3. ✅ **Test WebSocket**: Stream audio in real-time
4. 📝 **Build Frontend**: React/Vue/Svelte web UI
5. 🐳 **Containerize**: Create Docker image
6. 🚀 **Deploy**: AWS, GCP, or on-premise server

---

## 📚 Resources

- FastAPI Docs: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/
- WebSockets: https://fastapi.tiangolo.com/advanced/websockets/
- Uvicorn: https://www.uvicorn.org/
- Example clients in `examples/` folder

---

## Questions?

Refer to the examples:
- REST API: See Swagger UI at `/docs`
- WebSocket: Run `python examples/websocket_client.py`
- Curl: See `examples/curl_examples.sh`

---

**🎉 Migration Complete! Your Shady project is now a modern FastAPI service.**
