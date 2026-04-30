"""FastAPI application for Shady audio processing service."""

import json
import time
import asyncio
from typing import Dict
import time

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.config import COOLDOWN, MIC_RATE, FRAME_MS, VAD_RATE, TRIGGER_LIMIT
from backend.schemas import (
    AudioProcessingResponse,
    LLMResponse,
    HealthResponse,
    ErrorResponse,
)
from backend.services import (
    VADService,
    SpeakerVerificationService,
    TranscriptionService,
    LLMService,
    TTSService,
)

# Initialize FastAPI app
app = FastAPI(
    title="Shady Audio Processing API",
    description="Real-time audio processing with VAD, speaker verification, transcription, and LLM",
    version="1.0.0",
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
last_speech_time = None

# Initialize services
vad_service = VADService()
speaker_service = SpeakerVerificationService()

# Instantiate the remaining services
transcription_service = TranscriptionService() # Loads Whisper or similar
llm_service = LLMService()                     # Connects to GPT/Claude
tts_service = TTSService()                     # Initializes ElevenLabs/local TTS
# Track cooldowns per lesson
lesson_cooldowns: Dict[str, float] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global vad_service, speaker_service, transcription_service, llm_service, tts_service

    try:
        vad_service = VADService()
        speaker_service = SpeakerVerificationService()
        transcription_service = TranscriptionService()
        llm_service = LLMService()
        tts_service = TTSService()
    except Exception as e:
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        models_loaded=all(
            [
                vad_service is not None,
                speaker_service is not None,
                transcription_service is not None,
                llm_service is not None,
                tts_service is not None,
            ]
        ),
        student_enrolled=speaker_service is not None
        and speaker_service.student_embedding is not None,
    )


@app.post(
    "/process-audio",
    response_model=AudioProcessingResponse,
    responses={400: {"model": ErrorResponse}},
)
async def process_audio_file(file: UploadFile = File(...)):
    """
    Process audio file through full pipeline.

    - Read audio file
    - Detect speech
    - Verify speaker
    - Transcribe
    - Process with LLM
    - Generate audio response
    """
    try:
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be audio format")

        # Read file
        audio_bytes = await file.read()

        # Transcribe
        text = transcription_service.transcribe_from_bytes(audio_bytes)
        if not text:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        # Speaker verification
        is_student = speaker_service.is_student_voice(audio_bytes)
        similarity = speaker_service.get_speaker_similarity(audio_bytes)

        # Process with LLM
        llm_result = llm_service.process_transcript(text)

        # Generate TTS if nudging
        audio_generated = False
        if llm_result.get("should_nudge") and llm_result.get("nudge"):
            nudge_text = llm_result["nudge"] + " WHY: " + (llm_result.get("why") or "")
            audio_generated = tts_service.speak(nudge_text)

        return AudioProcessingResponse(
            transcript=text,
            speaker_is_student=is_student,
            speaker_similarity=similarity,
            llm_response=LLMResponse(**llm_result),
            audio_generated=audio_generated,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/generated")
async def get_generated_audio():
    """Get last generated audio file."""
    try:
        return FileResponse(
            path=tts_service.output_path,
            media_type="audio/wav",
            filename="response.wav",
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail="No audio generated yet")

# WEBSOCKET ENDPOINT

@app.websocket("/ws/audio")
async def websocket_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.

    Expected message format:
    {
        "type": "audio_chunk",
        "data": {
            "audio": "<base64 encoded audio bytes>",
            "sample_rate": 16000
        }
    }

    Response format:
    {
        "type": "status|response|error",
        "data": {...}
    }
    """
    await websocket.accept()

    try:
        buf = []
        speech_frames = 0
        silence_frames = 0
        active = False

        while True:
            # Receive message
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json(
                    {"type": "status", "data": {"message": "Timeout - no data received"}}
                )
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "data": {"message": "Invalid JSON"}}
                )
                continue

            msg_type = data.get("type")

            if msg_type == "audio_chunk":
                audio_b64 = data.get("data", {}).get("audio")
                try:
                    import base64

                    audio_data = data.get("data", {})
                    audio_b64 = audio_data.get("audio")

                    if not audio_b64:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "data": {"message": "No audio data in chunk"},
                            }
                        )
                        continue

                    # Decode base64 audio
                    audio_bytes = base64.b64decode(audio_b64)
                    if len(audio_bytes) % 2 != 0:
                        # Remove the trailing odd byte so it doesn't crash the converter
                        audio_bytes = audio_bytes[:-1]
                    if not audio_bytes:
                        continue
                    
                    # Convert to numpy array
                    arr = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_tensor = torch.from_numpy(arr.astype(np.float32) / 32768.0)

                    # VAD detection
                    speech_prob = vad_service.detect_speech(audio_tensor)

                    if speech_prob > 0.5:
                        buf.append(audio_bytes)
                        speech_frames += 1
                        silence_frames = 0
                        active = True
                        print("user talking")
                        last_speech_time = time.time()

                        await websocket.send_json(
                            {
                                "type": "status",
                                "data": {
                                    "state": "listening",
                                    "speech_prob": speech_prob,
                                },
                            }
                        )

                    elif active:
                        buf.append(audio_bytes)
                        silence_frames += 1
                        now = time.time()

                        if last_speech_time is None:
                            last_speech_time = now

                        silence_duration = now - last_speech_time
                        print(f"Silence duration: {silence_duration:.2f}s | frames: {silence_frames}")

                        trigger_limit = 6 
                        # Check if silence is long enough to end utterance
                        if silence_frames > trigger_limit:  # ~1.5 seconds of silence
                            print("Silence threshold reached (time-based)")
                            print("user not talking for this time")
                            if speech_frames > 0.5:  # Minimum speech duration
                                await websocket.send_json(
                                    {
                                        "type": "status",
                                        "data": {"state": "processing"},
                                    }
                                )

                                # Combine audio chunks
                                full_audio = b"".join(buf)

                                # Speaker verification
                                print("checking is this student voice")
                                is_student = speaker_service.is_student_voice(full_audio)

                                if is_student:
                                    await websocket.send_json(
                                        {
                                            "type": "status",
                                            "data": {"message": "Student voice detected, skipping"},
                                        }
                                    )
                                    buf, speech_frames, silence_frames, active = [], 0, 0, False
                                    continue

                                # Transcribe
                                text = transcription_service.transcribe_from_bytes(
                                    full_audio
                                )

                                if not text:
                                    await websocket.send_json(
                                        {
                                            "type": "status",
                                            "data": {"message": "Could not transcribe"},
                                        }
                                    )
                                else:
                                    await websocket.send_json(
                                        {
                                            "type": "status",
                                            "data": {"state": "transcribed", "text": text},
                                        }
                                    )

                                    # Get speaker similarity
                                    similarity = (
                                        speaker_service.get_speaker_similarity(full_audio)
                                    )
                                    print("speaker similarity", similarity)
                                    # Process with LLM
                                    llm_result = llm_service.process_transcript(text)

                                    response_data = {
                                        "transcript": text,
                                        "speaker_similarity": similarity,
                                        "llm_response": llm_result,
                                    }

                                    # Check cooldown
                                    lesson_id = llm_result.get("lesson_id")
                                    if lesson_id:
                                        now = time.time()
                                        if (
                                            now - lesson_cooldowns.get(lesson_id, 0)
                                        ) < COOLDOWN:
                                            response_data["cooldown_active"] = True
                                            response_data["should_nudge"] = False

                                            await websocket.send_json(
                                                {
                                                    "type": "response",
                                                    "data": response_data,
                                                }
                                            )

                                            buf, speech_frames, silence_frames, active = (
                                                [],
                                                0,
                                                0,
                                                False,
                                            )
                                            continue

                                        lesson_cooldowns[lesson_id] = now

                                    # Generate TTS if nudging
                                    if llm_result.get("should_nudge") and llm_result.get(
                                        "nudge"
                                    ):
                                        nudge_text = (
                                            llm_result["nudge"]
                                            + " WHY: "
                                            + (llm_result.get("why") or "")
                                        )
                                        tts_service.speak(nudge_text)
                                        response_data["audio_generated"] = True

                                    await websocket.send_json(
                                        {"type": "response", "data": response_data}
                                    )
                            else:
                                print("Ignored short speech") 
                            buf, speech_frames, silence_frames, active = [], 0, 0, False

                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "data": {"message": str(e)}}
                    )

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "close":
                await websocket.send_json({"type": "status", "data": {"message": "Closing connection"}})
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json(
                {"type": "error", "data": {"message": f"Server error: {str(e)}"}}
            )
        except:
            pass


# ============================================================================
# ROOT ENDPOINT
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Shady Audio Processing API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "process_audio": "POST /process-audio (upload file)",
            "websocket": "WS /ws/audio",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
