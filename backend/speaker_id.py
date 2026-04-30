import torch
import torchaudio
import torchaudio.transforms as T

import os
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.dataio.dataio import read_audio
import torch.nn.functional as F

def verify_speaker(test_audio_path):
    model = get_verification_model()
    ref_embedding = get_student_embedding() # Ensure this is [1, 192]
    
    # 1. Load and ensure correct shape
    test_signal = read_audio(test_audio_path)
    if test_signal.ndim == 1:
        test_signal = test_signal.unsqueeze(0)
    
    # 2. Get embedding
    with torch.no_grad():
        test_embedding = model.encode_batch(test_signal)
    
    # 3. Manual Cosine Similarity (mimicking SpeechBrain internal)
    # SpeechBrain embeddings are usually [batch, time, feature] 
    # We flatten to [batch, feature]
    score = F.cosine_similarity(ref_embedding.flatten(), test_embedding.flatten(), dim=0)
    
    return score.item()

# ====================== GLOBAL CACHE ======================
_verification_model = None
_student_embedding = None
_threshold = 0.24   # Good starting threshold for ECAPA model

def get_verification_model():
    """Load SpeechBrain ECAPA model **strictly from local folder**."""
    global _verification_model

    if _verification_model is not None:
        return _verification_model

    model_dir = "backend/models/spkrec-ecapa-voxceleb"

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found at: {os.path.abspath(model_dir)}\n"
            "Please ensure you have downloaded the model correctly."
        )

    # Check if critical files exist
    required_files = ["hyperparams.yaml", "embedding_model.ckpt"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing critical model files: {missing}")

    try:
        if torch.cuda.is_available():
            run_opts = {"device": "cuda:0"}
        else:
            run_opts = {"device": "cpu"}

        # Force local loading - avoid any remote fetch
        _verification_model = SpeakerRecognition.from_hparams(
            source=model_dir,      # Local path
            savedir=model_dir,     # Local path
            run_opts=run_opts
        )

        return _verification_model

    except Exception as e:
        raise


import soundfile as sf
import torch
import numpy as np
from scipy.signal import resample

def get_student_embedding(force_reload=False):
    global _student_embedding

    if _student_embedding is not None and not force_reload:
        return _student_embedding

    voice_ref_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "audio", "my_voice_sample.wav")
    )

    print("Audio path:", voice_ref_path)

    try:
        if not os.path.exists(voice_ref_path):
            raise FileNotFoundError(f"Reference voice file not found: {voice_ref_path}")

        print("1 → loading audio via soundfile")

        signal, sr = sf.read(voice_ref_path)
        print(f"Loaded. Sample rate: {sr} Hz | Shape: {signal.shape}")

        # Convert to mono
        if len(signal.shape) > 1:
            signal = signal.mean(axis=1)

        target_sr = 16000

        if sr != target_sr:
            print(f"Resampling {sr} → {target_sr}")
            num_samples = int(len(signal) * target_sr / sr)
            signal = resample(signal, num_samples)
        else:
            print("Already 16kHz")

        # Convert to torch tensor [1, time]
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        print("Final tensor shape:", signal.shape)

        model = get_verification_model()

        embedding = model.encode_batch(signal)
        _student_embedding = embedding.squeeze(0).squeeze(0)

        print("Embedding shape:", _student_embedding.shape)

        return _student_embedding

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error:", e)
        return None
