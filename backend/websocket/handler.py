import asyncio
import json
import time
from fastapi import WebSocket, WebSocketDisconnect

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
    background_tasks = set()
    
    try:
        while True:
            try:
                # Receive either text (JSON control messages) or binary (audio data)
                ws_msg = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json(timeout_status())
                continue
            
            # Handle binary audio data
            if "bytes" in ws_msg:
                audio_bytes = ws_msg["bytes"]
                await handle_binary_audio(websocket, audio_bytes, state, audio_processor, pipeline, background_tasks)
            # Handle text/JSON messages
            elif "text" in ws_msg:
                try:
                    data = json.loads(ws_msg["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(invalid_json_error())
                    continue
                
                msg_type = data.get("type")
                
                if msg_type == "ping":
                    await websocket.send_json(pong_response())
                elif msg_type == "close":
                    await websocket.send_json(closing_status())
                    break
                elif msg_type == "config":
                    # Store client sample rate for proper resampling
                    client_sample_rate = data.get("sampleRate", 48000)
                    state.client_sample_rate = client_sample_rate
                    print(f"Client sample rate: {client_sample_rate}Hz")
                else:
                    await websocket.send_json(generic_error(f"Unknown message type: {msg_type}"))
            else:
                await websocket.send_json(generic_error("Invalid message format"))
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json(server_error(str(e)))
        except:
            pass
    finally:
        # Cancel any pending background tasks
        for task in background_tasks:
            task.cancel()

async def handle_audio_chunk(websocket, data, state, audio_processor, pipeline, background_tasks):
    """Handle JSON audio_chunk messages (legacy support with base64)."""
    audio_data = data.get("data", {})
    audio_b64 = audio_data.get("audio")
    # Use client sample rate from state, fallback to provided or default
    sample_rate = data.get("sample_rate", state.client_sample_rate)
    
    chunk_result, error = audio_processor.process_chunk(audio_b64, sample_rate)
    if error:
        await websocket.send_json(generic_error(error))
        return
    
    speech_prob = chunk_result["speech_prob"]
    audio_bytes = chunk_result["audio_bytes"]
    
    if speech_prob > 0.5:
        # If we're already processing, cancel it and start fresh
        if state.processing:
            print("New speech detected while processing - starting new utterance")
            state.cancel_current = True
            state.reset()
        
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
        
        # Use time-based silence detection - 0.7 seconds
        if silence_duration > 0.7:
            print("Silence threshold reached (0.7s)")
            
            if state.speech_frames > 0.5:
                await websocket.send_json(processing_status())
                full_audio = b"".join(state.buf)
                
                # Reset state immediately to allow new audio capture
                state.reset()
                
                # Process utterance in background
                task = asyncio.create_task(
                    process_and_respond(websocket, pipeline, full_audio, state)
                )
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
            else:
                print("Ignored short speech")
                state.reset()
        
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
        
        # Use time-based silence detection - 0.7 seconds
        if silence_duration > 0.7:
            print("Silence threshold reached (0.7s)")
            
            if state.speech_frames > 1:
                await websocket.send_json(processing_status())
                full_audio = b"".join(state.buf)
                
                # Reset state immediately to allow new audio capture
                state.reset()
                
                # Process utterance in background
                task = asyncio.create_task(
                    process_and_respond(websocket, pipeline, full_audio, state)
                )
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
            else:
                print("Ignored short speech")
                state.reset()

async def handle_binary_audio(websocket, audio_bytes, state, audio_processor, pipeline, background_tasks):
    """Handle raw binary audio data from WebSocket."""
    # Pass raw bytes and client sample rate to audio processor for resampling
    chunk_result, error = audio_processor.process_chunk(audio_bytes, state.client_sample_rate)
    if error:
        await websocket.send_json(generic_error(error))
        return
    
    speech_prob = chunk_result["speech_prob"]
    audio_bytes = chunk_result["audio_bytes"]
    
    if speech_prob > 0.5:
        # If we're already processing, cancel it and start fresh
        if state.processing:
            print("New speech detected while processing - starting new utterance")
            state.cancel_current = True
            state.reset()
        
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
        
        # Use time-based silence detection - 0.7 seconds
        if silence_duration > 0.7:
            print("Silence threshold reached (0.7s)")
            
            if state.speech_frames > 0.5:
                await websocket.send_json(processing_status())
                full_audio = b"".join(state.buf)
                
                # Reset state immediately to allow new audio capture
                state.reset()
                
                # Process utterance in background
                task = asyncio.create_task(
                    process_and_respond(websocket, pipeline, full_audio, state)
                )
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
            else:
                print("Ignored short speech")
                state.reset()

async def process_and_respond(websocket, pipeline, full_audio, state):
    try:
        pipeline_result = await pipeline.process_utterance(full_audio, state)
        
        if pipeline_result["is_student"]:
            await websocket.send_json(student_voice_skipping_status())
            return
        
        if not pipeline_result["transcript"]:
            await websocket.send_json(transcription_failed_status())
            return

        # 1. Send the JSON metadata first (Transcript & LLM text)
        await websocket.send_json(transcribed_status(pipeline_result["transcript"]))
        await websocket.send_json(response_message(pipeline_result["response_data"]))
        
        # 2. Start streaming the audio binary chunks immediately
        nudge_text = pipeline_result["response_data"].get("nudge", "")
        if nudge_text:
            print(f"🔊 Streaming TTS for: {nudge_text[:50]}...")
            chunk_count = 0
            try:
                async for chunk in pipeline.tts_service.stream_audio(nudge_text):
                    chunk_count += 1
                    # Send raw bytes. No JSON, No Base64.
                    await websocket.send_bytes(chunk)
                print(f"✅ TTS streaming complete. Sent {chunk_count} chunks.")
            except Exception as tts_error:
                print(f"❌ TTS Streaming Error: {tts_error}")
                import traceback
                traceback.print_exc()
        else:
            print("ℹ️ No nudge text to stream")

    except Exception as e:
        print(f"❌ Processing Error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json(generic_error(f"Processing error: {str(e)}"))
    finally:
        state.processing = False
