import asyncio
import io
import math
import os
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from scipy.signal import resample_poly
from transformers import AutoModelForRNNT, AutoProcessor


MODEL_ID = os.getenv("ASR_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
MAX_UPLOAD_BYTES = int(os.getenv("ASR_MAX_UPLOAD_BYTES", str(500 << 20)))
semaphore = asyncio.Semaphore(max(1, int(os.getenv("ASR_MAX_CONCURRENCY", "1"))))
processor = None
model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global processor, model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForRNNT.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    yield


app = FastAPI(title="Nemotron 3.5 ASR native API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


def decode_audio(data: bytes):
    try:
        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError(f"unsupported or invalid audio: {exc}") from exc
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    target_rate = int(processor.feature_extractor.sampling_rate)
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        samples = resample_poly(samples, target_rate // divisor, sample_rate // divisor)
    return np.ascontiguousarray(samples, dtype=np.float32), target_rate


def transcribe(data: bytes, language: str | None):
    samples, sample_rate = decode_audio(data)
    language = language if language and language.lower() not in {"none"} else "auto"
    inputs = processor(
        samples,
        sampling_rate=sample_rate,
        language=language,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, return_dict_in_generate=True)
    text = processor.decode(output.sequences, skip_special_tokens=True)
    if isinstance(text, list):
        text = text[0]
    return text, language


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model_name: str | None = Form(None, alias="model"),
    language: str | None = Form(None),
):
    if model_name and model_name not in {MODEL_ID, "nemotron-asr"}:
        raise HTTPException(400, f"unsupported model: {model_name}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "audio file is too large")
    if not data:
        raise HTTPException(400, "audio file is empty")
    try:
        async with semaphore:
            text, detected_language = await asyncio.to_thread(transcribe, data, language)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"transcription failed: {exc}") from exc
    return {"text": text, "language": detected_language}
