#!/usr/bin/env python3
"""Small OpenAI-style image facade for the SeedVR2 ComfyUI nodes."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict


COMFY_URL = "http://127.0.0.1:8189"
MODEL_ID = "seedvr2-3b-fp8"
DIT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE_MODEL = "ema_vae_fp16.safetensors"
INPUT_ROOT = Path("/opt/ComfyUI/input").resolve()
OUTPUT_ROOT = Path("/opt/ComfyUI/output").resolve()
upscale_lock = asyncio.Lock()


class UpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image: str
    model: str = MODEL_ID
    scale: int = 2
    seed: int | None = None
    response_format: str = "b64_json"
    output_format: str = "png"


app = FastAPI(title="SeedVR2 Image Upscale API")


def decode_image(encoded: str) -> Image.Image:
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image must be valid base64") from exc
    if len(raw) > 64 << 20:
        raise HTTPException(status_code=400, detail="image exceeds 64 MiB")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="image is not a valid image") from exc


def save_input(image: Image.Image) -> tuple[str, Path]:
    relative = f"seedvr2-api/{uuid.uuid4().hex}.png"
    path = INPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return relative, path


def workflow(image_name: str, short_edge: int, seed: int, prefix: str) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {
            "class_type": "SeedVR2LoadDiTModel",
            "inputs": {
                "model": DIT_MODEL, "device": "cuda:0", "blocks_to_swap": 0,
                "swap_io_components": False, "offload_device": "none",
                "cache_model": False, "attention_mode": "sdpa",
            },
        },
        "3": {
            "class_type": "SeedVR2LoadVAEModel",
            "inputs": {
                "model": VAE_MODEL, "device": "cuda:0",
                "encode_tiled": True, "encode_tile_size": 1024, "encode_tile_overlap": 128,
                "decode_tiled": True, "decode_tile_size": 1024, "decode_tile_overlap": 128,
                "tile_debug": "false", "offload_device": "none", "cache_model": False,
            },
        },
        "4": {
            "class_type": "SeedVR2VideoUpscaler",
            "inputs": {
                "image": ["1", 0], "dit": ["2", 0], "vae": ["3", 0], "seed": seed,
                "resolution": short_edge, "max_resolution": 4096, "batch_size": 1,
                "uniform_batch_size": False, "temporal_overlap": 0, "prepend_frames": 0,
                "color_correction": "lab", "input_noise_scale": 0.0,
                "latent_noise_scale": 0.0, "offload_device": "cpu", "enable_debug": True,
            },
        },
        "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["4", 0]}},
    }


async def comfy_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"{COMFY_URL}/object_info")
        return response.is_success
    except httpx.HTTPError:
        return False


async def wait_for_output(client: httpx.AsyncClient, prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 60 * 60
    while time.monotonic() < deadline:
        response = await client.get(f"{COMFY_URL}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json().get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI upscale failed: {status.get('messages', [])}")
            for output in history.get("outputs", {}).values():
                images = output.get("images", [])
                if images:
                    return images[0]
        await asyncio.sleep(0.5)
    raise TimeoutError("SeedVR2 upscale timed out")


async def execute_workflow(graph: dict[str, Any]) -> str:
    output: dict[str, Any] | None = None
    try:
        async with httpx.AsyncClient(timeout=60 * 60) as client:
            submitted = await client.post(f"{COMFY_URL}/prompt", json={"prompt": graph})
            submitted.raise_for_status()
            body = submitted.json()
            if body.get("node_errors"):
                raise RuntimeError(f"invalid SeedVR2 workflow: {body['node_errors']}")
            output = await wait_for_output(client, body["prompt_id"])
            viewed = await client.get(f"{COMFY_URL}/view", params=output)
            viewed.raise_for_status()
            return base64.b64encode(viewed.content).decode("ascii")
    finally:
        if output is not None:
            candidate = (OUTPUT_ROOT / output["subfolder"] / output["filename"]).resolve()
            if candidate.is_relative_to(OUTPUT_ROOT):
                candidate.unlink(missing_ok=True)


@app.get("/health")
async def health() -> dict[str, str]:
    if not await comfy_ready():
        raise HTTPException(status_code=503, detail="SeedVR2 runtime is starting")
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


@app.post("/v1/images/upscale")
async def upscale(request: UpscaleRequest) -> dict[str, Any]:
    if request.model != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"model mismatch: {request.model}")
    if request.scale < 2 or request.scale > 4:
        raise HTTPException(status_code=400, detail="scale must be between 2 and 4")
    if request.response_format != "b64_json" or request.output_format != "png":
        raise HTTPException(status_code=400, detail="only b64_json PNG output is supported")
    image = decode_image(request.image)
    target_width, target_height = image.width * request.scale, image.height * request.scale
    if target_width > 4096 or target_height > 4096:
        raise HTTPException(status_code=400, detail="output must not exceed 4096 pixels on either edge")
    image_name, input_path = save_input(image)
    seed = request.seed if request.seed is not None and request.seed >= 0 else secrets.randbits(32)
    prefix = f"seedvr2-api/{uuid.uuid4().hex}"
    try:
        async with upscale_lock:
            encoded = await execute_workflow(workflow(image_name, min(target_width, target_height), seed, prefix))
        return {"created": int(time.time()), "data": [{"b64_json": encoded}], "seed": seed}
    finally:
        input_path.unlink(missing_ok=True)
