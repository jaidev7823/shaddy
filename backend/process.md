# Shady Audio Processing - Complete Flow Documentation

## Overview
Shady is a real-time English coaching system that listens to conversations via WebSocket, detects speech, verifies speaker identity, transcribes audio, generates coaching nudges via LLM, and speaks responses using Edge TTS (Microsoft's neural TTS).

**Key Feature**: Parallel processing - new speech can be captured and processed while previous utterances are still being transcribed/processed in the background.

---

## Phase 1: Application Startup (`app.py`)

### 1. Configuration Loading (`config.py`)
- Loads environment variables (`.env` file)
- Sets device: CUDA if available, else CPU
- Defines constants: `MIC_RATE=16000`, `VAD_RATE=16000`, `FRAME_MS=32`, `COOLDOWN=10s`
- Reads LLM provider: Ollama (default) or Gemini
- Sets paths: `LESSONS_PATH`, `SAMPLE_VOICE_STUDENT`, `VOICE_REF`

### 2. Model Initialization (`models.py`)
- **Whisper Model**: Loads Faster Whisper medium model (auto-downloads if not cached)
- **TTS**: Uses Edge TTS (`edge-tts` package) - no local model needed

### 3. Service Initialization (`app.py` lines 39-44)
```
vad_service = VADService()
speaker_service = SpeakerVerificationService()
transcription_service = TranscriptionService()
llm_service = LLMService()
tts_service = TTSService()
```

### 4. Startup Event (`app.py` `startup_event()`)
- Re-initializes all services (ensures fresh state)
- Services are globally available to all routes

---

## Phase 2: WebSocket Connection (`websocket/handler.py`)

### 1. Client Connects to `/ws/audio`
```
Frontend → WebSocket connect → /ws/audio
```

### 2. Connection Accepted (`handler.py`)
```python
await websocket.accept()
```

### 3. Session State Created (`handler.py`)
```python
state = SessionState()  # from websocket/state.py
```
State tracks:
- `buf`: Audio chunks buffer
- `speech_frames`: Count of speech frames
- `silence_frames`: Count of silence frames
- `active`: Whether currently in speech mode
- `last_speech_time`: Timestamp of last speech
- `processing`: Whether an utterance is being processed in background
- `cancel_current`: Flag to cancel current processing if new speech starts

### 4. Audio & Pipeline Processors Initialized (`handler.py`)
```python
audio_processor = AudioProcessor(vad_service)
pipeline = Pipeline(speaker_service, transcription_service, llm_service, tts_service)
background_tasks = set()
```

---

## Phase 3: Message Processing Loop (`handler.py` `websocket_audio_stream()`)

### 1. Receive Message (30s timeout)
```python
msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
```

### 2. Parse JSON
```python
data = json.loads(msg)
msg_type = data.get("type")
```

### 3. Route by Message Type

| Type | Handler | Action |
|------|---------|--------|
| `audio_chunk` | `handle_audio_chunk()` | Process audio (potentially start background task) |
| `ping` | Direct | Return `{"type": "pong"}` |
| `close` | Direct | Send closing status, break loop |
| Other | Direct | Return error |

---

## Phase 4: Audio Chunk Processing (`handler.py` `handle_audio_chunk()`)

### Step 1: Extract Audio Data
```python
audio_data = data.get("data", {})
audio_b64 = audio_data.get("audio")
```

### Step 2: Process Chunk (`audio_processor.py` `process_chunk()`)

#### 2a. Base64 Decode
```python
audio_bytes = base64.b64decode(audio_b64)
if len(audio_bytes) % 2 != 0:
    audio_bytes = audio_bytes[:-1]  # Fix odd byte
```

#### 2b. Convert to Tensor
```python
arr = np.frombuffer(audio_bytes, dtype=np.int16)
audio_tensor = torch.from_numpy(arr.astype(np.float32) / 32768.0)
```

#### 2c. VAD Detection (`vad_service.py` `detect_speech()`)
```python
speech_prob = self.model(audio_tensor, 16000).item()
```
- Uses **Silero VAD** model
- Pads/trims tensor to 512 samples (Silero requirement)
- Returns probability (0.0 to 1.0)

### Step 3: Speech Detection Decision

#### If `speech_prob > 0.5` (Speech Detected)
```python
# If we're already processing, cancel it and start fresh
if state.processing:
    print("New speech detected while processing - starting new utterance")
    state.cancel_current = True
    state.reset()

state.buf.append(audio_bytes)
state.speech_frames += 1
state.active = True
state.last_speech_time = time.time()
```
Send status: `{"type": "status", "data": {"state": "listening", "speech_prob": ...}}`

#### If `speech_prob <= 0.5` AND `state.active` (Silence During Speech)
```python
state.buf.append(audio_bytes)
state.silence_frames += 1
```
Check silence duration...

### Step 4: Silence Threshold Check (`TRIGGER_LIMIT = 800 // 32 = 25 frames`)
```python
if state.silence_frames > TRIGGER_LIMIT:
    # End of utterance detected
    await websocket.send_json(processing_status())
    full_audio = b"".join(state.buf)
    
    # Reset state immediately to allow new audio capture
    state.reset()
    
    # Process utterance in background (non-blocking)
    task = asyncio.create_task(
        process_and_respond(websocket, pipeline, full_audio, state)
    )
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
```

**Key Change**: State is reset immediately after silence threshold, allowing new speech to be captured while the previous utterance is still being processed in the background.

---

## Phase 5: Background Processing (`handler.py` `process_and_respond()`)

### Async Function for Parallel Processing
```python
async def process_and_respond(websocket, pipeline, full_audio, state):
    try:
        pipeline_result = await pipeline.process_utterance(full_audio, state)
        
        if pipeline_result["is_student"]:
            await websocket.send_json(student_voice_skipping_status())
            return
        
        if not pipeline_result["transcript"]:
            await websocket.send_json(transcription_failed_status())
            return
        
        await websocket.send_json(transcribed_status(pipeline_result["transcript"]))
        await websocket.send_json(response_message(pipeline_result["response_data"]))
    except Exception as e:
        await websocket.send_json(generic_error(f"Processing error: {str(e)}"))
    finally:
        state.processing = False
```

---

## Phase 6: Full Utterance Processing (`pipeline.py` `process_utterance()`)

### Step 1: Check for Cancellation
```python
if state and state.cancel_current:
    state.processing = False
    state.cancel_current = False
    return result  # Early exit if new speech detected
```

### Step 2: Process Audio (passed as parameter)
```python
# full_audio is passed directly to process_utterance() as parameter
# No need to access state.buf since state is reset immediately after silence detection
```

### Step 3: Speaker Verification (`speaker_service.py`)

#### 3a. Check if Student Voice
```python
is_student = speaker_service.is_student_voice(full_audio)
```
- Loads **SpeechBrain ECAPA-TDNN** model (lazy load)
- Converts audio bytes → tensor via `_audio_bytes_to_tensor()`
- Computes embedding using `model.encode_batch()`
- Compares with `student_embedding` using cosine similarity
- Returns `True` if `similarity > threshold` (default threshold from `speaker_id.py`)

**If student voice detected** → Send skip message, return

#### 3b. Get Speaker Similarity Score
```python
similarity = speaker_service.get_speaker_similarity(full_audio)
```

### Step 4: Transcription (`transcription_service.py` `transcribe_from_bytes()`)

#### 4a. Save to Temporary WAV
```python
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    with wave.open(f, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(MIC_RATE)
        w.writeframes(audio_bytes)
```

#### 4b. Transcribe with Faster Whisper
```python
segs, _ = self.whisper.transcribe(
    tmp,
    language=None,  # Auto-detect
    task="translate",  # Translate to English
    vad_filter=True,
    temperature=0.0,
    beam_size=5
)
text = " ".join(s.text for s in segs).strip()
```

**If transcription fails** → Send error, return

### Step 5: LLM Processing (`llm_service.py` → `llm.py`)

#### 5a. Build Prompt (`prompt.py` `build_prompt()`)
```
### ROLE
You are a silent English coach in the student's ear...
### TOPICS
[Loaded from lessons/lessons.json]
### OUTPUT FIELDS
1. "answer" — A natural spoken reply
2. "why" — What concept the sentence teaches
3. "lesson_id" — Topic ID
```

#### 5b. Call LLM (`llm.py` `ask_llm()`)
```python
def ask_llm(transcript):
    return ask_ollama(transcript) if PROVIDER == "ollama" else ask_gemini(transcript)
```

**Ollama Path** (`ask_ollama()`):
```python
response = ollama.generate(model=OLLAMA_MODEL, prompt=build_prompt(transcript))
return _parse_ollama(response["response"])
```

**Gemini Path** (`ask_gemini()`):
```python
client = genai.Client(api_key=GEMINI_API_KEY)
response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=build_prompt(transcript),
    config={"response_mime_type": "application/json"}
)
return _parse_ollama(response.text)
```

#### 5c. Parse Response (`_parse_ollama()`)
```python
data = json.loads(raw)
return {
    "should_nudge": bool(answer and answer.upper() != "NONE"),
    "lesson_id": data.get("lesson_id"),
    "nudge": data.get("answer"),
    "why": data.get("why")
}
```

### Step 6: Cooldown Check (`websocket/cooldown.py`)

```python
lesson_id = llm_result.get("lesson_id")
if lesson_id:
    if cooldown_manager.is_active(lesson_id):
        # Skip nudge, still return response with "cooldown_active": True
    cooldown_manager.update(lesson_id)  # Reset cooldown timer
```

- Cooldown period: `COOLDOWN = 10 seconds`
- Per-lesson cooldown tracking
- Uses global `cooldown_manager` singleton

### Step 7: Text-to-Speech (`tts_service.py` → Edge TTS)

#### 7a. Check if Nudge Needed
```python
if llm_result.get("should_nudge") and nudge:
    nudge_text = nudge + " WHY: " + (llm_result.get("why") or "")
    await tts_service.speak(nudge_text)
```

#### 7b. Generate Audio with Edge TTS (`tts_service.py`)
```python
async def speak(self, text: str) -> bool:
    await self._generate(text)
    subprocess.run(["aplay", "-q", str(self.output_path)], check=True)
    return True

async def _generate(self, text: str):
    communicate = edge_tts.Communicate(text, voice="en-IN-NeerjaNeural")
    await communicate.save(str(self.output_path))
```

**Edge TTS** (in `models.py`):
- Uses Microsoft's cloud-based neural TTS
- Voice: `en-IN-NeerjaNeural` (Indian English, natural sounding)
- No local model needed - generates audio files directly

---

## Phase 7: Response Sent to Frontend

### Response Structure
```json
{
    "type": "response",
    "data": {
        "transcript": "What were you doing that day?",
        "speaker_similarity": 0.6543,
        "llm_response": {
            "should_nudge": true,
            "lesson_id": "GEN-01",
            "nudge": "I was helping my friend with some work.",
            "why": "Past continuous for ongoing past action"
        },
        "audio_generated": true,
        "cooldown_active": false
    }
}
```

---

## File Responsibility Summary (Sequential Order)

| Order | File | Responsibility |
|-------|------|----------------|
| 1 | `config.py` | Load environment, define constants |
| 2 | `models.py` | Load Whisper model (TTS is cloud-based via edge-tts) |
| 3 | `speaker_id.py` | Load SpeechBrain ECAPA model + student embedding |
| 4 | `app.py` | Initialize FastAPI, services, define routes |
| 5 | `services/vad_service.py` | VADService: Silero VAD detection |
| 6 | `services/vad_service.py` | SpeakerVerificationService: SpeechBrain speaker ID |
| 7 | `services/transcription_service.py` | Faster Whisper transcription |
| 8 | `prompt.py` | Build LLM prompt with lessons context |
| 9 | `llm.py` | Call Ollama/Gemini, parse JSON response |
| 10 | `services/llm_service.py` | LLMService wrapper |
| 11 | `services/tts_service.py` | Edge TTS generation + playback (async) |
| 12 | `websocket/state.py` | Session state management (with `processing`/`cancel_current` flags) |
| 13 | `websocket/messages.py` | JSON response builders |
| 14 | `websocket/cooldown.py` | Per-lesson cooldown tracking |
| 15 | `websocket/audio_processor.py` | Audio chunk decode + VAD |
| 16 | `websocket/pipeline.py` | Full utterance processing orchestration (async, cancellation support) |
| 17 | `websocket/handler.py` | WebSocket connection + message routing (with background tasks) |

---

## Data Flow Diagram

```
Frontend (Mic)
    ↓ WebSocket /ws/audio
    ↓ audio_chunk {audio: base64}
handler.py:websocket_audio_stream()
    ↓
handle_audio_chunk()
    ↓
audio_processor.py:process_chunk()
    ↓ base64 decode → numpy → torch tensor
    ↓
vad_service.py:detect_speech() → speech_prob
    ↓
[If speech_prob > 0.5] → Buffer audio, send "listening" status
    ↓                    [If speech starts while processing → cancel_current=True]
[If silence > TRIGGER_LIMIT] → Combine buffered audio
    ↓
[state.reset() immediately - frees up for new capture]
    ↓
asyncio.create_task(process_and_respond()) ← BACKGROUND TASK STARTS
    ↓                                           ↓
pipeline.py:process_utterance(full_audio, state)   [New audio captured here]
    ↓
[Check state.cancel_current → early exit if new speech]
    ↓
speaker_service.py:is_student_voice() → Skip if student
    ↓
transcription_service.py:transcribe_from_bytes() → text
    ↓
llm_service.py:process_transcript(text)
    ↓
llm.py:ask_llm() → Ollama/Gemini → JSON response
    ↓
cooldown.py: Check/update lesson cooldown
    ↓
[If should_nudge] → tts_service.py:speak() [ASYNC]
    ↓
edge_tts.Communicate → save WAV → aplay
    ↓
process_and_respond() sends WebSocket responses:
    ↓
WebSocket: Send {"type": "transcribed", "data": {...}}
WebSocket: Send {"type": "response", "data": {...}}
    ↓
Frontend receives nudge + plays audio

Note: Multiple utterances can be in different pipeline stages concurrently!
```

---

## Key Design Patterns

1. **Lazy Loading**: Speaker verification model loads on first use (`_get_model()`)
2. **Global Singleton**: `cooldown_manager` shared across connections
3. **Session State**: Each WebSocket connection gets its own `SessionState`
4. **Service Layer**: `services/` contains business logic, `websocket/` handles real-time concerns
5. **Pipeline Pattern**: `pipeline.py` orchestrates the full utterance processing sequence
6. **Parallel Processing**: Background tasks (`asyncio.create_task`) allow new speech to be captured while previous utterances are processing
7. **Cancellation Support**: `state.cancel_current` flag allows new speech to cancel ongoing processing

---

## Parallel Processing Flow

```
Time →

[Utterance 1 starts] 
    ↓
[Silence detected for Utterance 1]
    ↓
[State reset - ready for new input]
    ↓
[Background Task 1 started: Transcribe → LLM → TTS]
    ↓                                       [Utterance 2 starts - captured normally]
    ↓                                       [Silence detected for Utterance 2]
    ↓                                       [Background Task 2 started]
    ↓                                       [Utterance 3 starts...]
[Task 1 completes → Send response]
[Task 2 completes → Send response]
```

**Benefits**:
- No blocking: User can speak again immediately after pausing
- Multiple utterances can be processed concurrently
- Old processing is cancelled if new speech starts (`cancel_current` flag)
- Better user experience - no "blocked" feeling
