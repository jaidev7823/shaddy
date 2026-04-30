import torch
import os
from speechbrain.inference.speaker import SpeakerRecognition

# ====================== GLOBAL CACHE ======================
_verification_model = None
_student_embedding = None
_threshold = 0.75   # Good starting threshold for ECAPA model


def get_verification_model():
    """Load SpeechBrain ECAPA model **strictly from local folder**."""
    global _verification_model

    if _verification_model is not None:
        return _verification_model

    model_dir = "backend/models/spkrec-ecapa-voxceleb"

    if not os.path.exists(model_dir):
        print("file not fount")
        raise FileNotFoundError(
            f"Model directory not found at: {os.path.abspath(model_dir)}\n"
            "Please ensure you have downloaded the model correctly."
        )

    # Check if critical files exist
    required_files = ["hyperparams.yaml", "embedding_model.ckpt"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing critical model files: {missing}")

    print(f"🔄 Loading SpeechBrain ECAPA model from local directory:")
    print(f"   → {os.path.abspath(model_dir)}")

    try:
        if torch.cuda.is_available():
            run_opts = {"device": "cuda:0"}
            print("   Device: GPU (cuda:0)")
        else:
            run_opts = {"device": "cpu"}
            print("   Device: CPU")

        # Force local loading - avoid any remote fetch
        _verification_model = SpeakerRecognition.from_hparams(
            source=model_dir,      # Local path
            savedir=model_dir,     # Local path
            run_opts=run_opts
        )

        print("✅ SpeechBrain ECAPA-TDNN model loaded successfully from local folder!")
        return _verification_model

    except Exception as e:
        print(f"❌ Failed to load SpeechBrain model: {e}")
        print("   Tip: Make sure 'hyperparams.yaml' and 'embedding_model.ckpt' exist inside the folder.")
        raise


from speechbrain.dataio.dataio import read_audio

def get_student_embedding(force_reload=False):
    """Load student reference voice embedding using SpeechBrain-compatible loader."""
    global _student_embedding

    if _student_embedding is not None and not force_reload:
        return _student_embedding

    voice_ref_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "audio", "my_voice_sample.wav")
    )

    print("Resolved path:", voice_ref_path)

    try:
        if not os.path.exists(voice_ref_path):
            raise FileNotFoundError(f"Reference voice file not found: {voice_ref_path}")

        # 🔥 Use SpeechBrain loader instead of torchaudio
        signal = read_audio(voice_ref_path)  # shape: [time]

        # Convert to [1, time]
        signal = signal.unsqueeze(0)

        model = get_verification_model()

        embedding = model.encode_batch(signal)
        _student_embedding = embedding.squeeze(0).squeeze(0)

        print(f"✅ Student voice reference loaded | Embedding shape: {_student_embedding.shape}")
        return _student_embedding

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Failed to enroll student voice: {e}")
        return None
