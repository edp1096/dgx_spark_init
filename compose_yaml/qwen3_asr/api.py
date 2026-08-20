import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from qwen_asr import Qwen3ASRModel


MODEL_ID = os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGNER_ID = os.getenv("ASR_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_NEW_TOKENS", "256"))
MAX_UPLOAD_BYTES = int(os.getenv("ASR_MAX_UPLOAD_BYTES", str(500 << 20)))
semaphore = asyncio.Semaphore(max(1, int(os.getenv("ASR_MAX_CONCURRENCY", "1"))))
engine = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        forced_aligner=ALIGNER_ID,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
        max_inference_batch_size=1,
        max_new_tokens=MAX_NEW_TOKENS,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    yield


app = FastAPI(title="Qwen3-ASR timestamp API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "forced_aligner": ALIGNER_ID}


def transcribe(path: str, language: str | None, prompt: str | None):
    results = engine.transcribe(
        audio=path,
        context=prompt or "",
        language=language,
        return_time_stamps=True,
    )
    if not results:
        raise ValueError("recognition returned no result")
    result = results[0]
    timestamps = []
    if result.time_stamps is not None:
        timestamps = [
            {"text": item.text, "start": float(item.start_time), "end": float(item.end_time)}
            for item in result.time_stamps.items
        ]
    return {"text": result.text, "language": result.language, "timestamps": timestamps}


async def save_upload(file: UploadFile, path: Path):
    written = 0
    with path.open("wb") as destination:
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "audio file is too large")
            destination.write(chunk)
    if written == 0:
        raise HTTPException(400, "audio file is empty")


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model_name: str | None = Form(None, alias="model"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
):
    if model_name and model_name not in {MODEL_ID, "Qwen/Qwen3-ASR-1.7B-hf", "qwen3-asr"}:
        raise HTTPException(400, f"unsupported model: {model_name}")
    normalized_language = None if not language or language.lower() in {"auto", "none"} else language
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="qwen3-asr-") as directory:
        path = Path(directory) / ("input" + suffix)
        await save_upload(file, path)
        try:
            async with semaphore:
                return await asyncio.to_thread(transcribe, str(path), normalized_language, prompt)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"transcription failed: {exc}") from exc
