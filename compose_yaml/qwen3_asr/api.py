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
from transformers import AutoModelForMultimodalLM, AutoProcessor


MODEL_ID = os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B-hf")
MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_NEW_TOKENS", "2048"))
MAX_UPLOAD_BYTES = int(os.getenv("ASR_MAX_UPLOAD_BYTES", str(500 << 20)))
semaphore = asyncio.Semaphore(max(1, int(os.getenv("ASR_MAX_CONCURRENCY", "1"))))
processor = None
model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global processor, model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    yield


app = FastAPI(title="Qwen3-ASR native API", lifespan=lifespan)


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
    return np.ascontiguousarray(samples, dtype=np.float32)


def transcribe(data: bytes, language: str | None, prompt: str | None):
    samples = decode_audio(data)
    kwargs = {"audio": samples}
    if language and language.lower() not in {"auto", "none"}:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt
    inputs = processor.apply_transcription_request(**kwargs).to(model.device, model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.decode(generated_ids, return_format="parsed")[0]


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model_name: str | None = Form(None, alias="model"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
):
    if model_name and model_name not in {MODEL_ID, "qwen3-asr"}:
        raise HTTPException(400, f"unsupported model: {model_name}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "audio file is too large")
    if not data:
        raise HTTPException(400, "audio file is empty")
    try:
        async with semaphore:
            result = await asyncio.to_thread(transcribe, data, language, prompt)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"transcription failed: {exc}") from exc
    return {
        "text": result.get("transcription", ""),
        "language": result.get("language"),
    }
