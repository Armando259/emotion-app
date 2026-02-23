import os
import warnings
import logging

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import soundfile as sf
from transformers import pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

TRANSFORMER_MODEL_ID = "svantip123/TransformerAudioEmotion"
_model = None


def _read_audio(audio_path: str) -> dict:
    """Čita WAV direktno u numpy array — bez ffmpeg"""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return {"array": audio, "sampling_rate": sr}


def _get_model():
    global _model
    if _model is None:
        _model = pipeline(
            "audio-classification",
            model=TRANSFORMER_MODEL_ID,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _model


def predict_transformer_audio(audio_path: str) -> dict:
    try:
        audio_input = _read_audio(audio_path)
        model = _get_model()
        results = model(audio_input)
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return {
            "status": "success",
            "model": "TransformerAudioEmotion",
            "type": "audio",
            "top_emotion": results[0]["label"],
            "top_score": round(results[0]["score"] * 100, 2),
            "all_predictions": [
                {"label": r["label"], "score": round(r["score"] * 100, 2)}
                for r in results
            ],
        }
    except Exception as e:
        return {
            "status": "error",
            "model": "TransformerAudioEmotion",
            "error": str(e),
        }
