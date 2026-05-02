"""FastAPI application for Shady audio processing service."""

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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
from backend.websocket.handler import websocket_audio_stream as ws_handler

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
vad_service = VADService()
speaker_service = SpeakerVerificationService()
transcription_service = TranscriptionService()
llm_service = LLMService()
tts_service = TTSService()

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
async def websocket_audio_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming."""
    await ws_handler(
        websocket,
        vad_service,
        speaker_service,
        transcription_service,
        llm_service,
        tts_service,
    )


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
