#!/usr/bin/env python3
"""OpenAI-compatible API for SANA-Sprint 0.6B with an NF4 text encoder."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import secrets
import time

import torch
from diffusers import SanaSprintPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, ImageFilter
from transformers import AutoModel, BitsAndBytesConfig

from editing import SprintMaskedEditor, build_outpaint_canvas


MODEL = os.getenv(
    "SANA_MODEL",
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
)
MODEL_ID = "sana-sprint-0.6b-nf4"
generation_lock = asyncio.Lock()


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1)
    model: str = MODEL_ID
    n: int = Field(default=1, ge=1, le=1)
    size: str = "1024x1024"
    seed: int | None = None
    steps: int = Field(default=2, ge=1, le=4)
    response_format: str = "b64_json"


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (AttributeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="size must be WIDTHxHEIGHT") from exc
    if not (256 <= width <= 2048 and 256 <= height <= 2048):
        raise HTTPException(status_code=400, detail="width and height must be between 256 and 2048")
    if width % 64 or height % 64:
        raise HTTPException(status_code=400, detail="width and height must be multiples of 64")
    return width, height


def load_pipeline() -> SanaSprintPipeline:
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    text_encoder = AutoModel.from_pretrained(
        MODEL,
        subfolder="text_encoder",
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    pipeline = SanaSprintPipeline.from_pretrained(
        MODEL,
        text_encoder=text_encoder,
        dtype=torch.bfloat16,
    )
    pipeline.transformer.to("cuda")
    pipeline.vae.to("cuda")
    pipeline.vae.enable_tiling(
        tile_sample_min_height=512,
        tile_sample_min_width=512,
        tile_sample_stride_height=448,
        tile_sample_stride_width=448,
    )
    return pipeline


pipeline = load_pipeline()
editor = SprintMaskedEditor(pipeline)
app = FastAPI(title="SANA-Sprint 0.6B NF4 API")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "text_encoder": "gemma2-2b-nf4",
        "vae_tiling": True,
        "capabilities": ["text-to-image", "inpaint-experimental", "outpaint-experimental"],
    }


@app.get("/v1/models")
def models() -> dict[str, object]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


def encode_result(image: Image.Image, seed: int, started: float) -> dict[str, object]:
    torch.cuda.synchronize()
    output = io.BytesIO()
    image.save(output, format="PNG")
    return {
        "created": int(time.time()),
        "data": [{"b64_json": base64.b64encode(output.getvalue()).decode("ascii")}],
        "seed": seed,
        "metrics": {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        },
    }


def generate(request: ImageRequest) -> dict[str, object]:
    if request.model not in {MODEL_ID, MODEL}:
        raise HTTPException(status_code=400, detail=f"unsupported model: {request.model}")
    if request.response_format != "b64_json":
        raise HTTPException(status_code=400, detail="only b64_json response_format is supported")
    width, height = parse_size(request.size)
    seed = request.seed if request.seed is not None and request.seed >= 0 else secrets.randbits(63)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        image = pipeline(
            prompt=request.prompt.strip(),
            width=width,
            height=height,
            num_inference_steps=request.steps,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
    return encode_result(image, seed, started)


@app.post("/v1/images/generations")
async def image_generation(request: ImageRequest) -> dict[str, object]:
    async with generation_lock:
        return await asyncio.to_thread(generate, request)


async def read_image(upload: UploadFile, mode: str) -> Image.Image:
    payload = await upload.read()
    if not payload or len(payload) > 32 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="image must be between 1 byte and 32 MiB")
    try:
        opened = Image.open(io.BytesIO(payload))
        opened.load()
        return opened.convert(mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc


def masked_edit(
    source: Image.Image,
    mask: Image.Image,
    prompt: str,
    size: str,
    steps: int,
    seed: int | None,
) -> dict[str, object]:
    width, height = parse_size(size)
    actual_seed = seed if seed is not None and seed >= 0 else secrets.randbits(63)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    image = editor.edit(
        source=source,
        mask=mask,
        prompt=prompt.strip(),
        width=width,
        height=height,
        steps=steps,
        seed=actual_seed,
    )
    return encode_result(image, actual_seed, started)


@app.post("/v1/images/edits")
async def image_edit(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(..., min_length=1),
    size: str = Form("1024x1024"),
    steps: int = Form(2, ge=2, le=4),
    seed: int | None = Form(None),
    feather: int = Form(16, ge=0, le=128),
) -> dict[str, object]:
    source_image = await read_image(image, "RGB")
    mask_image = await read_image(mask, "L")
    if feather:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(feather))
    async with generation_lock:
        return await asyncio.to_thread(masked_edit, source_image, mask_image, prompt, size, steps, seed)


@app.post("/v1/images/outpaint")
async def image_outpaint(
    image: UploadFile = File(...),
    prompt: str = Form(..., min_length=1),
    left: int = Form(0, ge=0, le=1024),
    right: int = Form(0, ge=0, le=1024),
    top: int = Form(0, ge=0, le=1024),
    bottom: int = Form(0, ge=0, le=1024),
    overlap: int = Form(64, ge=0, le=256),
    feather: int = Form(24, ge=0, le=128),
    steps: int = Form(2, ge=2, le=4),
    seed: int | None = Form(None),
) -> dict[str, object]:
    source_image = await read_image(image, "RGB")
    if left + right + top + bottom == 0:
        raise HTTPException(status_code=400, detail="at least one outpaint margin must be positive")
    try:
        canvas = build_outpaint_canvas(
            source_image,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            overlap=overlap,
            feather=feather,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    size = f"{canvas.image.width}x{canvas.image.height}"
    parse_size(size)
    async with generation_lock:
        return await asyncio.to_thread(masked_edit, canvas.image, canvas.mask, prompt, size, steps, seed)
