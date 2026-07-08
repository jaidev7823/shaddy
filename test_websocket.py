#!/usr/bin/env python3
"""Test WebSocket endpoint with TTS streaming"""
import asyncio
import websockets
import json
import base64

async def test():
    uri = "ws://localhost:8000/ws/audio"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        # Send a ping
        await websocket.send(json.dumps({"type": "ping"}))
        response = await websocket.recv()
        print(f"Ping response: {response}")
        
        # Send a fake audio chunk (base64 encoded silence)
        # In reality, this would be actual audio data
        fake_audio = b'\x00' * 4096  # 4096 bytes of silence
        audio_b64 = base64.b64encode(fake_audio).decode('utf-8')
        
        await websocket.send(json.dumps({
            "type": "audio_chunk",
            "data": {
                "audio": audio_b64,
                "sample_rate": 16000
            }
        }))
        
        print("Sent audio chunk, waiting for response...")
        
        # Wait for responses
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                
                if isinstance(response, bytes):
                    print(f"Received binary data: {len(response)} bytes")
                else:
                    print(f"Received JSON: {response[:200]}...")
        except asyncio.TimeoutError:
            print("Timeout waiting for response")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
