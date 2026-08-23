#!/usr/bin/env python3
"""OpenAI-compatible facade for Z-Image Turbo NVFP4 on ComfyUI."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict


COMFY_PORT = int(os.getenv("COMFY_PORT", "8290"))
COMFY_URL = f"http://127.0.0.1:{COMFY_PORT}"
UNLOAD_AFTER_GENERATION = os.getenv("UNLOAD_AFTER_GENERATION", "true").lower() in {
    "1",
    "true",
    "yes",
}
MODEL_ID = "z-image-turbo-nvfp4"
MODEL_ALIASES = {MODEL_ID, "Tongyi-MAI/Z-Image-Turbo"}
DIFFUSION_MODEL = "z_image_turbo_nvfp4.safetensors"
TEXT_ENCODER = "qwen_3_4b_fp4_mixed.safetensors"
VAE = "ae.safetensors"
CONTROLNET = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"
OUTPUT_ROOT = Path("/opt/ComfyUI/output").resolve()
INPUT_ROOT = Path("/opt/ComfyUI/input").resolve()
generation_lock = asyncio.Lock()


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str
    model: str = MODEL_ID
    n: int = 1
    size: str = "1024x1024"
    seed: int | None = None
    response_format: str = "b64_json"
    control_image: str | None = None
    control_strength: float = 0.75
    control_strategy: str = "split4"


app = FastAPI(title="Z-Image Turbo NVFP4 API")


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="size must be WIDTHxHEIGHT") from exc
    if not (512 <= width <= 2048 and 512 <= height <= 2048):
        raise HTTPException(status_code=400, detail="width and height must be between 512 and 2048")
    if width % 16 or height % 16:
        raise HTTPException(status_code=400, detail="width and height must be multiples of 16")
    return width, height


def workflow(prompt: str, width: int, height: int, seed: int, prefix: str) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "lumina2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": 9,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }


def control_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    input_name: str,
    strength: float,
    strategy: str,
) -> dict[str, Any]:
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "lumina2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "10": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "11": {
            "class_type": "Canny",
            "inputs": {"image": ["10", 0], "low_threshold": 0.4, "high_threshold": 0.8},
        },
        "12": {"class_type": "ModelPatchLoader", "inputs": {"name": CONTROLNET}},
        "13": {
            "class_type": "ZImageFunControlnet",
            "inputs": {
                "model": ["1", 0],
                "model_patch": ["12", 0],
                "vae": ["3", 0],
                "image": ["11", 0],
                "strength": strength,
            },
        },
        "14": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["13", 0], "shift": 3.0},
        },
        "15": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": 3.0},
        },
    }

    common = {
        "positive": ["4", 0],
        "negative": ["5", 0],
        "steps": 8,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
    }
    if strategy == "full8":
        graph["7"] = {
            "class_type": "KSampler",
            "inputs": {
                **common,
                "model": ["14", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "denoise": 1.0,
            },
        }
        samples = ["7", 0]
    else:
        graph["7"] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                **common,
                "model": ["14", 0],
                "latent_image": ["6", 0],
                "add_noise": "enable",
                "noise_seed": seed,
                "start_at_step": 0,
                "end_at_step": 4,
                "return_with_leftover_noise": "enable",
            },
        }
        graph["16"] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                **common,
                "model": ["15", 0],
                "latent_image": ["7", 0],
                "add_noise": "disable",
                "noise_seed": seed,
                "start_at_step": 4,
                "end_at_step": 8,
                "return_with_leftover_noise": "disable",
            },
        }
        samples = ["16", 0]

    graph["8"] = {"class_type": "VAEDecode", "inputs": {"samples": samples, "vae": ["3", 0]}}
    graph["9"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": prefix, "images": ["8", 0]},
    }
    return graph


def save_control_image(encoded: str) -> tuple[str, Path]:
    payload = encoded.split(",", 1)[1] if encoded.startswith("data:") and "," in encoded else encoded
    try:
        raw = base64.b64decode(payload, validate=True)
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError("control image exceeds 32 MiB")
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
            image.load()
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid control_image: {exc}") from exc

    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    name = f"zimage-control-{uuid.uuid4().hex}.png"
    path = INPUT_ROOT / name
    image.save(path, format="PNG")
    return name, path


async def comfy_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"{COMFY_URL}/object_info")
        return response.is_success
    except httpx.HTTPError:
        return False


async def wait_for_output(client: httpx.AsyncClient, prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 30 * 60
    while time.monotonic() < deadline:
        response = await client.get(f"{COMFY_URL}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json().get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI generation failed: {status.get('messages', [])}")
            images = history.get("outputs", {}).get("9", {}).get("images", [])
            if images:
                return images[0]
        await asyncio.sleep(0.25)
    raise TimeoutError("image generation timed out")


async def execute_workflow(graph: dict[str, Any]) -> str:
    image: dict[str, Any] | None = None
    try:
        async with httpx.AsyncClient(timeout=30 * 60) as client:
            submitted = await client.post(f"{COMFY_URL}/prompt", json={"prompt": graph})
            submitted.raise_for_status()
            body = submitted.json()
            if body.get("node_errors"):
                raise RuntimeError(f"invalid workflow: {body['node_errors']}")
            image = await wait_for_output(client, body["prompt_id"])
            viewed = await client.get(f"{COMFY_URL}/view", params=image)
            viewed.raise_for_status()
            return base64.b64encode(viewed.content).decode("ascii")
    finally:
        if image is not None:
            candidate = (OUTPUT_ROOT / image["subfolder"] / image["filename"]).resolve()
            if candidate.is_relative_to(OUTPUT_ROOT):
                candidate.unlink(missing_ok=True)


async def release_models() -> None:
    if not UNLOAD_AFTER_GENERATION:
        return
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{COMFY_URL}/free",
                json={"unload_models": True, "free_memory": True},
            )
            response.raise_for_status()
    except httpx.HTTPError:
        # Cache release is best effort and must not fail a completed request.
        pass


@app.get("/health")
async def health() -> dict[str, str]:
    if not await comfy_ready():
        raise HTTPException(status_code=503, detail="NVFP4 runtime is starting")
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


@app.post("/v1/images/generations")
async def generate(request: ImageRequest) -> dict[str, Any]:
    if request.model not in MODEL_ALIASES:
        raise HTTPException(status_code=400, detail=f"model mismatch: {request.model}")
    if request.n != 1:
        raise HTTPException(status_code=400, detail="only n=1 is supported")
    if request.response_format != "b64_json":
        raise HTTPException(status_code=400, detail="only b64_json is supported")
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    width, height = parse_size(request.size)
    seed = request.seed if request.seed is not None and request.seed >= 0 else secrets.randbits(63)
    prefix = f"zimage-api/{uuid.uuid4().hex}"

    if not 0.0 <= request.control_strength <= 2.0:
        raise HTTPException(status_code=400, detail="control_strength must be between 0 and 2")
    if request.control_strategy not in {"full8", "split4"}:
        raise HTTPException(status_code=400, detail="control_strategy must be full8 or split4")

    input_path: Path | None = None
    if request.control_image:
        input_name, input_path = save_control_image(request.control_image)
        graph = control_workflow(
            prompt,
            width,
            height,
            seed,
            prefix,
            input_name,
            request.control_strength,
            request.control_strategy,
        )
    else:
        graph = workflow(prompt, width, height, seed, prefix)

    async with generation_lock:
        try:
            try:
                encoded = await execute_workflow(graph)
            except (httpx.HTTPError, KeyError, RuntimeError, TimeoutError) as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if input_path is not None:
                input_path.unlink(missing_ok=True)
            await release_models()

    return {"created": int(time.time()), "data": [{"b64_json": encoded}]}
