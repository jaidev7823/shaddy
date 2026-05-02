from typing import Optional

def timeout_status():
    return {"type": "status", "data": {"message": "Timeout - no data received"}}

def invalid_json_error():
    return {"type": "error", "data": {"message": "Invalid JSON"}}

def no_audio_data_error():
    return {"type": "error", "data": {"message": "No audio data in chunk"}}

def listening_status(speech_prob: float):
    return {
        "type": "status",
        "data": {
            "state": "listening",
            "speech_prob": speech_prob,
        },
    }

def processing_status():
    return {"type": "status", "data": {"state": "processing"}}

def student_voice_skipping_status():
    return {
        "type": "status",
        "data": {"message": "Student voice detected, skipping"},
    }

def transcription_failed_status():
    return {"type": "status", "data": {"message": "Could not transcribe"}}

def transcribed_status(text: str):
    return {
        "type": "status",
        "data": {"state": "transcribed", "text": text},
    }

def response_message(data: dict):
    return {"type": "response", "data": data}

def pong_response():
    return {"type": "pong"}

def closing_status():
    return {"type": "status", "data": {"message": "Closing connection"}}

def server_error(message: str):
    return {"type": "error", "data": {"message": f"Server error: {message}"}}

def generic_error(message: str):
    return {"type": "error", "data": {"message": message}}
