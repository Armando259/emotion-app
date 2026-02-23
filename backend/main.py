import asyncio
import concurrent.futures
import os
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from models import predict_pulse
from transformer_model import predict_transformer_audio
from lstm_model import predict_lstm_audio

app = FastAPI(title="Emotion Recognition API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/lstm-inspect")
def lstm_inspect():
    import torch
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="svantip123/LstmAudioEmotion", filename="lstm_best.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    
    model_state = state["model_state_dict"]
    config      = state.get("config", {})
    id2label    = state.get("id2label", {})
    
    return {
        "model_keys_and_shapes": {
            k: list(v.shape) for k, v in model_state.items() if hasattr(v, "shape")
        },
        "config": config,
        "id2label": id2label,
    }

@app.post("/api/predict")
async def predict_all(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "rec")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        loop = asyncio.get_event_loop()
        results = await asyncio.gather(
            loop.run_in_executor(executor, predict_transformer_audio, tmp_path),
            loop.run_in_executor(executor, predict_lstm_audio, tmp_path),
            loop.run_in_executor(executor, predict_pulse, tmp_path),
        )
        return {"status": "success", "results": list(results)}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/predict/transformer")
async def predict_transformer(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "rec")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, predict_transformer_audio, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/predict/pulse")
async def predict_pulse_endpoint(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "rec")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, predict_pulse, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/predict/lstm")
async def predict_lstm(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "rec")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, predict_lstm_audio, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
