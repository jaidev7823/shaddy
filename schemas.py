"""Pydantic models for API requests and responses."""

from typing import Optional

from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    """Request for transcription."""

    text: str = Field(..., min_length=1, description="Text to transcribe")


class LLMResponse(BaseModel):
    """LLM processing response."""

    should_nudge: bool = Field(..., description="Whether to provide a nudge")
    lesson_id: Optional[str] = Field(None, description="Lesson ID if applicable")
    nudge: Optional[str] = Field(None, description="The nudge message")
    why: Optional[str] = Field(None, description="Explanation for the nudge")


class AudioProcessingResponse(BaseModel):
    """Full audio processing response."""

    transcript: str = Field(..., description="Transcribed text")
    speaker_is_student: bool = Field(..., description="Whether speaker is the student")
    speaker_similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Speaker similarity score"
    )
    llm_response: LLMResponse
    audio_generated: bool = Field(..., description="Whether audio was generated")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether all models are loaded")
    student_enrolled: bool = Field(..., description="Whether student is enrolled")


class WebSocketMessage(BaseModel):
    """WebSocket message format."""

    type: str = Field(
        ..., description="Message type: 'audio_chunk', 'status', 'response', 'error'"
    )
    data: Optional[dict] = Field(None, description="Message data")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
