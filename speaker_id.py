import numpy as np
import librosa
from config import VOICE_REF

_student_embedding = None
_threshold = 0.82

def get_embedding(audio_data, sr=16000):
    y = audio_data.astype(np.float32) / 32768.0
    # Use more MFCCs and add delta features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # Concatenate mean and std for more discriminative embedding
    embedding = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        delta.mean(axis=1),
        delta2.mean(axis=1)
    ])
    return embedding

def get_student_embedding():
    global _student_embedding
    if _student_embedding is not None:
        return _student_embedding
    
    try:
        import soundfile as sf
        y, sr = sf.read(VOICE_REF)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        _student_embedding = get_embedding(y)
        print(f"Student voice enrolled. Embedding shape: {_student_embedding.shape}")
        return _student_embedding
    except Exception as e:
        print(f"Warning: Failed to load student voice: {e}")
        return None

def is_student_voice(audio_bytes, student_embedding, threshold=None):
    if student_embedding is None:
        return False
    
    try:
        y = np.frombuffer(audio_bytes, dtype=np.int16)
        embedding = get_embedding(y)
        
        # Cosine similarity
        similarity = np.dot(embedding, student_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(student_embedding) + 1e-6
        )
        
        thresh = threshold or _threshold
        print(f"  [Speaker similarity: {similarity:.3f}, threshold: {thresh}]")
        return similarity > thresh
    except Exception as e:
        print(f"Speaker verification error: {e}")
        return False
