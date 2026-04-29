#!/bin/bash
# FastAPI Migration - Example Commands

echo "🚀 Shady FastAPI - Usage Examples"
echo "=================================="
echo ""

# Health check
echo "1️⃣  Health Check:"
echo "curl -X GET http://localhost:8000/health | jq ."
echo ""

# Process audio file
echo "2️⃣  Process Audio File (HTTP):"
echo "curl -X POST -F 'file=@audio_sample.wav' http://localhost:8000/process-audio | jq ."
echo ""

# Get generated audio
echo "3️⃣  Get Generated Response Audio:"
echo "curl -X GET http://localhost:8000/audio/generated --output response.wav"
echo ""

# Interactive testing
echo "4️⃣  Interactive Testing:"
echo "   - Open browser: http://localhost:8000/docs"
echo "   - Try endpoints in Swagger UI"
echo ""

# WebSocket testing (Python)
echo "5️⃣  WebSocket Client (Python):"
echo "   See: examples/websocket_client.py"
echo ""

echo "✨ Start server with:"
echo "   python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000"
