"""WebSocket client example for real-time audio streaming."""

import asyncio
import base64
import json
import sys
from pathlib import Path
from backend.config import CHUNK_SIZE

import websockets


async def stream_audio_file(file_path: str, chunk_size=CHUNK_SIZE):
    """Stream audio file through WebSocket."""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    if not file_path.suffix.lower() == ".wav":
        print("⚠️  Warning: File is not .wav format")

    print(f"📁 Reading file: {file_path}")
    audio_data = file_path.read_bytes()
    print(f"📦 File size: {len(audio_data)} bytes")

    uri = "ws://localhost:8000/ws/audio"
    print(f"🔗 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")

            # Stream audio in chunks
            print("📤 Streaming audio...")
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                audio_b64 = base64.b64encode(chunk).decode("utf-8")

                message = {
                    "type": "audio_chunk",
                    "data": {"audio": audio_b64, "sample_rate": 16000},
                }

                await websocket.send(json.dumps(message))

                # Show progress
                progress = min(i + chunk_size, len(audio_data))
                print(f"  Sent {progress}/{len(audio_data)} bytes", end="\r")

                # Wait a bit to simulate real-time streaming
                await asyncio.sleep(0.01)

            print("\n✅ Audio streaming complete")

            # Send close message
            await websocket.send(json.dumps({"type": "close"}))

            # Receive responses
            print("\n📥 Responses:")
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    print(json.dumps(data, indent=2))
            except asyncio.TimeoutError:
                print("⏱️  No more messages (timeout)")

    except ConnectionRefusedError:
        print("❌ Connection refused. Is the server running?")
        print("   Start with: python -m uvicorn app:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")


async def interactive_mode():
    """Interactive WebSocket mode."""
    uri = "ws://localhost:8000/ws/audio"

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            print("Commands: 'ping', 'close', or base64-encoded audio")
            print("")

            async def send_messages():
                while True:
                    msg = await asyncio.get_event_loop().run_in_executor(
                        None, input, "Send: "
                    )

                    if msg.lower() == "ping":
                        await websocket.send(json.dumps({"type": "ping"}))
                    elif msg.lower() == "close":
                        await websocket.send(json.dumps({"type": "close"}))
                        break
                    else:
                        await websocket.send(msg)

            async def receive_messages():
                try:
                    while True:
                        response = await websocket.recv()
                        print(f"Received: {response}")
                except Exception:
                    pass

            await asyncio.gather(send_messages(), receive_messages())

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Stream file
        asyncio.run(stream_audio_file(sys.argv[1]))
    else:
        # Interactive mode
        print("🎙️  WebSocket Audio Client")
        print("===========================")
        print("")
        print("Usage:")
        print("  python examples/websocket_client.py <audio_file.wav>")
        print("  python examples/websocket_client.py  (interactive mode)")
        print("")

        try:
            asyncio.run(interactive_mode())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
