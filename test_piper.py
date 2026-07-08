#!/usr/bin/env python3
"""Test Piper TTS streaming"""
import sys
sys.path.insert(0, '/home/jaidev/Work/shady/backend')

try:
    from piper import PiperVoice
    print("✅ Piper imported successfully")
    
    # Load voice
    voice = PiperVoice.load("backend/models/en_US-lessac-medium.onnx")
    print("✅ Voice loaded successfully")
    
    # Test synthesize
    print("Testing synthesize...")
    result = voice.synthesize("Hello, world!")
    print(f"Result type: {type(result)}")
    
    if hasattr(result, '__next__'):
        print("It's a generator")
        chunks = []
        for i, chunk in enumerate(result):
            chunks.append(len(chunk))
            if i < 3:
                print(f"  Chunk {i}: {len(chunk)} bytes")
        print(f"Total chunks: {len(chunks)}")
        print(f"Chunk sizes: {chunks[:10]}...")
    elif isinstance(result, bytes):
        print(f"It's bytes, length: {len(result)}")
        # Check if it's WAV or raw PCM
        if result[:4] == b'RIFF':
            print("It's a WAV file")
        else:
            print("It's raw PCM")
    else:
        print(f"Unknown type: {result}")
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Piper might not be installed. Try: pip install piper-tts")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
