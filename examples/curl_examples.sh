#!/bin/bash
# Curl examples for Shady FastAPI

BASE_URL="http://localhost:8000"

echo "🎙️  Shady FastAPI - Curl Examples"
echo "=================================="
echo ""

# Health check
echo "1. Health Check"
echo "==============="
echo "curl -X GET $BASE_URL/health | jq ."
echo ""
curl -X GET "$BASE_URL/health" | jq . 2>/dev/null || echo "⚠️  Server not running"
echo ""
echo ""

# API docs
echo "2. Interactive API Documentation"
echo "================================="
echo "Open in browser: $BASE_URL/docs"
echo ""
echo ""

# Process audio file
echo "3. Process Audio File"
echo "====================="
echo "Requires: test_audio.wav in current directory"
echo ""

if [ -f "test_audio.wav" ]; then
    echo "curl -X POST -F 'file=@test_audio.wav' $BASE_URL/process-audio | jq ."
    echo ""
    curl -X POST -F "file=@test_audio.wav" "$BASE_URL/process-audio" 2>/dev/null | jq . || echo "Error processing file"
else
    echo "⚠️  test_audio.wav not found"
fi
echo ""
echo ""

# Get generated audio
echo "4. Get Generated Response Audio"
echo "================================"
echo "curl -X GET $BASE_URL/audio/generated --output response.wav"
echo ""
echo "Or stream it:"
echo "curl -X GET $BASE_URL/audio/generated | aplay"
echo ""
echo ""

# Using Python requests (alternative)
echo "5. Using Python Requests"
echo "========================"
echo ""
cat << 'EOF'
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Upload and process audio
with open("test_audio.wav", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/process-audio", files=files)
    print(response.json())

# Get generated audio
response = requests.get("http://localhost:8000/audio/generated")
with open("response.wav", "wb") as f:
    f.write(response.content)
EOF

echo ""
echo "6. Tips"
echo "======"
echo "- Start server: python -m uvicorn app:app --reload"
echo "- Interactive docs: http://localhost:8000/docs"
echo "- WebSocket client: python examples/websocket_client.py <audio.wav>"
echo "- View logs: tail -f shady.log"
