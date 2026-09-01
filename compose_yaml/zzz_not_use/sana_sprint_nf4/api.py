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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModel, BitsAndBytesConfig


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
app = FastAPI(title="SANA-Sprint 0.6B NF4 API")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "text_encoder": "gemma2-2b-nf4",
        "vae_tiling": True,
    }


@app.get("/v1/models")
def models() -> dict[str, object]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


def generate(request: ImageRequest) -> dict[str, object]:
    if request.model not in {MODEL_ID, MODEL}:
        raise HTTPException(status_code=400, detail=f"unsupported model: {request.model}")
    if request.response_format != "b64_json":
        raise HTTPException(status_code=400, detail="only b64_json response_format is supported")
    width, height = parse_size(request.size)
    seed = request.seed if request.seed is not None and request.seed >= 0 else secrets.randbits(63)
    with torch.inference_mode():
        image = pipeline(
            prompt=request.prompt.strip(),
            width=width,
            height=height,
            num_inference_steps=request.steps,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
    output = io.BytesIO()
    image.save(output, format="PNG")
    return {
        "created": int(time.time()),
        "data": [{"b64_json": base64.b64encode(output.getvalue()).decode("ascii")}],
        "seed": seed,
    }


@app.post("/v1/images/generations")
async def image_generation(request: ImageRequest) -> dict[str, object]:
    async with generation_lock:
        return await asyncio.to_thread(generate, request)
