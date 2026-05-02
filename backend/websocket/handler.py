import asyncio
import json
import time
from fastapi import WebSocket

from backend.config import TRIGGER_LIMIT
from backend.websocket.state import SessionState
from backend.websocket.audio_processor import AudioProcessor
from backend.websocket.pipeline import Pipeline
from backend.websocket.messages import (
    timeout_status,
    invalid_json_error,
    generic_error,
    listening_status,
    processing_status,
    student_voice_skipping_status,
    transcription_failed_status,
    transcribed_status,
    response_message,
    pong_response,
    closing_status,
    server_error,
)

async def websocket_audio_stream(
    websocket: WebSocket,
    vad_service,
    speaker_service,
    transcription_service,
    llm_service,
    tts_service,
):
    await websocket.accept()
    state = SessionState()
    audio_processor = AudioProcessor(vad_service)
    pipeline = Pipeline(speaker_service, transcription_service, llm_service, tts_service)
    
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json(timeout_status())
                continue
            
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await websocket.send_json(invalid_json_error())
                continue
            
            msg_type = data.get("type")
            
            if msg_type == "audio_chunk":
                await handle_audio_chunk(websocket, data, state, audio_processor, pipeline)
            elif msg_type == "ping":
                await websocket.send_json(pong_response())
            elif msg_type == "close":
                await websocket.send_json(closing_status())
                break
            else:
                await websocket.send_json(generic_error(f"Unknown message type: {msg_type}"))
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json(server_error(str(e)))
        except:
            pass

async def handle_audio_chunk(websocket, data, state, audio_processor, pipeline):
    audio_data = data.get("data", {})
    audio_b64 = audio_data.get("audio")
    
    chunk_result, error = audio_processor.process_chunk(audio_b64)
    if error:
        await websocket.send_json(generic_error(error))
        return
    
    speech_prob = chunk_result["speech_prob"]
    audio_bytes = chunk_result["audio_bytes"]
    
    if speech_prob > 0.5:
        state.buf.append(audio_bytes)
        state.speech_frames += 1
        state.silence_frames = 0
        state.active = True
        print("user talking")
        state.last_speech_time = time.time()
        await websocket.send_json(listening_status(speech_prob))
    
    elif state.active:
        state.buf.append(audio_bytes)
        state.silence_frames += 1
        now = time.time()
        
        if state.last_speech_time is None:
            state.last_speech_time = now
        
        silence_duration = now - state.last_speech_time
        print(f"Silence duration: {silence_duration:.2f}s | frames: {state.silence_frames}")
        
        if state.silence_frames > 2:
            print("Silence threshold reached")
            
            if state.speech_frames > 0.5:
                await websocket.send_json(processing_status())
                full_audio = b"".join(state.buf)
                pipeline_result = pipeline.process_utterance(full_audio)
                
                if pipeline_result["is_student"]:
                    await websocket.send_json(student_voice_skipping_status())
                    state.reset()
                    return
                
                if not pipeline_result["transcript"]:
                    await websocket.send_json(transcription_failed_status())
                    state.reset()
                    return
                
                await websocket.send_json(transcribed_status(pipeline_result["transcript"]))
                await websocket.send_json(response_message(pipeline_result["response_data"]))
            else:
                print("Ignored short speech")
            
            state.reset()
