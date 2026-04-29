"""FastAPI application for Shady audio processing service."""

import json
import time
import asyncio
from typing import Dict

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import COOLDOWN, MIC_RATE, FRAME_MS, VAD_RATE
from schemas import (
    AudioProcessingResponse,
    LLMResponse,
    HealthResponse,
    ErrorResponse,
)
from services import (
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

# Initialize services
vad_service = None
speaker_service = None
transcription_service = None
llm_service = None
tts_service = None

# Track cooldowns per lesson
lesson_cooldowns: Dict[str, float] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global vad_service, speaker_service, transcription_service, llm_service, tts_service

    print("🚀 Initializing services...")

    try:
        vad_service = VADService()
        print("✓ VAD service initialized")

        speaker_service = SpeakerVerificationService()
        print("✓ Speaker verification service initialized")

        transcription_service = TranscriptionService()
        print("✓ Transcription service initialized")

        llm_service = LLMService()
        print("✓ LLM service initialized")

        tts_service = TTSService()
        print("✓ TTS service initialized")

        print("✅ All services ready!")

    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise


# ============================================================================
# HTTP ENDPOINTS
# ============================================================================


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

        print(f"Transcript: {text}")
        print(f"Speaker is student: {is_student}")
        print(f"Similarity: {similarity:.3f}")

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
        print(f"Error processing audio: {e}")
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


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================


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
    print(f"🔗 WebSocket connection established")

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

                        # Check if silence is long enough to end utterance
                        if silence_frames > 1500 // FRAME_MS:  # ~1.5 seconds of silence
                            if speech_frames > 12:  # Minimum speech duration

                                await websocket.send_json(
                                    {
                                        "type": "status",
                                        "data": {"state": "processing"},
                                    }
                                )

                                # Combine audio chunks
                                full_audio = b"".join(buf)

                                # Speaker verification
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

                            # Reset for next utterance
                            buf, speech_frames, silence_frames, active = [], 0, 0, False

                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    await websocket.send_json(
                        {"type": "error", "data": {"message": str(e)}}
                    )

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "close":
                await websocket.send_json({"type": "status", "data": {"message": "Closing connection"}})
                break

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
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
