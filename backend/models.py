import os
import re
import warnings
import logging

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

import torch
import numpy as np
import soundfile as sf
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    pipeline,
)
from transformers.utils import logging as hf_logging
from peft import PeftConfig, PeftModel

hf_logging.set_verbosity_error()

# ─── Cache ────────────────────────────────────────────────────────────────────

_models = {}

PULSE_MODEL_ID = "drPantagana/PULSE_emotion"

PULSE_ID2LABEL = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "love",
    4: "surprise",
    5: "neutral",
}
PULSE_LABEL2ID = {v: k for k, v in PULSE_ID2LABEL.items()}


# ─── Halucinacija filter ──────────────────────────────────────────────────────

def _is_hallucination(text: str) -> bool:
    """Detektira Whisper halucinacije — previše simbola, premalo slova"""
    if not text or len(text.strip()) < 3:
        return True

    # Ako više od 15% znakova su simboli — halucinacija
    special = sum(1 for c in text if c in '$%&@#!*^~`|\\')
    ratio = special / max(len(text), 1)
    if ratio > 0.15:
        return True

    # Mora imati barem 3 normalna slova
    letters = re.findall(r'[a-zA-ZÀ-žА-яа-я]', text)
    if len(letters) < 3:
        return True

    return False


# ─── Audio helper ─────────────────────────────────────────────────────────────

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


# ─── Model loaderi ────────────────────────────────────────────────────────────

def _get_whisper():
    if "whisper" not in _models:
        _models["whisper"] = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=0 if torch.cuda.is_available() else -1,
        )
    return _models["whisper"]


def _get_pulse():
    if "pulse" not in _models:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        peft_config = PeftConfig.from_pretrained(PULSE_MODEL_ID)

        base_model = RobertaForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=6,
            id2label=PULSE_ID2LABEL,
            label2id=PULSE_LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        model = PeftModel.from_pretrained(base_model, PULSE_MODEL_ID)
        model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(PULSE_MODEL_ID)
        _models["pulse"] = (model, tokenizer, device)

    return _models["pulse"]


# ─── Predikcija ───────────────────────────────────────────────────────────────

def predict_pulse(audio_path: str) -> dict:
    try:
        # Korak 1: audio → numpy → Whisper → tekst
        audio_input = _read_audio(audio_path)
        whisper = _get_whisper()
        transcription = whisper(audio_input)["text"].strip()

        # ── Filter halucinacija ──
        if not transcription or _is_hallucination(transcription):
            return {
                "status": "error",
                "model": "PULSE_emotion",
                "error": "Govor nije prepoznat — snimi čišći audio s govorom",
                "transcribed_text": None,
            }

        # Korak 2: tekst → emocija (PULSE LoRA)
        model, tokenizer, device = _get_pulse()
        inputs = tokenizer(
            transcription,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        results = sorted(
            [
                {
                    "label": PULSE_ID2LABEL[i],
                    "score": round(float(probs[i]) * 100, 2),
                }
                for i in range(len(probs))
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        return {
            "status": "success",
            "model": "PULSE_emotion",
            "type": "text",
            "transcribed_text": transcription,
            "top_emotion": results[0]["label"],
            "top_score": results[0]["score"],
            "all_predictions": results,
        }

    except Exception as e:
        return {
            "status": "error",
            "model": "PULSE_emotion",
            "error": str(e),
            "transcribed_text": None,
        }
