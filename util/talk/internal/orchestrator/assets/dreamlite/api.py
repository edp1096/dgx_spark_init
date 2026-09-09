"""DreamLite Mobile's four-step generation and single-reference editing API."""
import asyncio
import base64
import binascii
import io
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import torch
from diffusers import DreamLiteMobilePipeline
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

MODEL = "carlofkl/DreamLite-mobile"
REVISION = "6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4"
pipe = None
inference_lock = threading.Lock()


def load_model():
    global pipe
    pipe = DreamLiteMobilePipeline.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
    ).to("cuda")


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(load_model)
    yield


app = FastAPI(lifespan=lifespan)


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["carlofkl/DreamLite-mobile"] = MODEL
    prompt: str = Field(min_length=1, max_length=16000)
    size: str = "1024x1024"
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    n: Literal[1] = 1
    response_format: Literal["b64_json"] = "b64_json"
    output_format: Literal["png"] = "png"
    source_image: str | None = Field(default=None, max_length=32 * 1024 * 1024)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "steps": 4,
            "operations": ["generate", "identity_edit"]}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "local"}]}


def generate(request):
    match = re.fullmatch(r"(\d+)x(\d+)", request.size)
    if not match:
        raise HTTPException(400, "size must be WIDTHxHEIGHT")
    width, height = map(int, match.groups())
    if any(value < 512 or value > 1024 or value % 16 for value in (width, height)):
        raise HTTPException(400, "trial dimensions must be 512..1024 and divisible by 16")
    if not request.prompt.strip():
        raise HTTPException(400, "prompt must not be blank")
    source = None
    if request.source_image is not None:
        try:
            header, encoded = request.source_image.split(",", 1)
            if not header.startswith("data:image/") or not header.endswith(";base64"):
                raise ValueError("expected an image data URL")
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > 16_777_216:
                    raise ValueError("source image exceeds 16 megapixels")
                source = image.convert("RGB")
        except (ValueError, binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, "invalid source_image") from exc
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(409, "image generation is already running")
    try:
        seed = request.seed if request.seed is not None else secrets.randbelow(2**63)
        with torch.inference_mode():
            result = pipe(prompt=request.prompt, image=source, width=width, height=height,
                          num_inference_steps=4,
                          generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        output = io.BytesIO()
        result.save(output, format="PNG")
        return {"created": int(time.time()), "seed": seed,
                "data": [{"b64_json": base64.b64encode(output.getvalue()).decode("ascii")}]}
    finally:
        inference_lock.release()


@app.post("/v1/images/generations")
async def generations(request: Generation):
    return await asyncio.to_thread(generate, request)
