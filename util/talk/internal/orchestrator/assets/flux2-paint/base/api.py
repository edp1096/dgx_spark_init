#!/usr/bin/env python3
"""Small OpenAI-compatible facade for a loopback-only ComfyUI runtime."""

from __future__ import annotations

import asyncio
import base64
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict


COMFY_PORT = os.getenv("COMFY_PORT", "8188")
COMFY_URL = os.getenv("COMFY_URL", f"http://127.0.0.1:{COMFY_PORT}")
MODEL_ID = "flux2-klein-4b-nvfp4"
MODEL_ALIASES = {MODEL_ID, "black-forest-labs/FLUX.2-klein-4b-nvfp4"}
DIFFUSION_MODEL = "flux-2-klein-4b-nvfp4.safetensors"
TEXT_ENCODER = "qwen_3_4b_flux2.safetensors"
VAE = "flux2-vae.safetensors"
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


app = FastAPI(title="FLUX.2 Klein 4B NVFP4 API")


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="size must be WIDTHxHEIGHT") from exc
    if not (256 <= width <= 2048 and 256 <= height <= 2048):
        raise HTTPException(status_code=400, detail="width and height must be between 256 and 2048")
    if width % 16 or height % 16:
        raise HTTPException(status_code=400, detail="width and height must be multiples of 16")
    return width, height


def workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    references: list[str] | None = None,
) -> dict[str, Any]:
    graph: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": width, "height": height}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["4", 0]}},
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["9", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["5", 0],
            },
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["11", 0]}},
    }
    conditioning: list[Any] = ["4", 0]
    for index, reference in enumerate(references or []):
        node = 20 + index * 4
        graph[str(node)] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        graph[str(node + 1)] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": [str(node), 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "center",
            },
        }
        graph[str(node + 2)] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": [str(node + 1), 0], "vae": ["3", 0]},
        }
        graph[str(node + 3)] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": conditioning, "latent": [str(node + 2), 0]},
        }
        conditioning = [str(node + 3), 0]
    graph["9"]["inputs"]["conditioning"] = conditioning
    return graph


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
                messages = status.get("messages", [])
                raise RuntimeError(f"ComfyUI generation failed: {messages}")
            images = history.get("outputs", {}).get("12", {}).get("images", [])
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


def request_seed(value: int | None) -> int:
    return value if value is not None and value >= 0 else secrets.randbits(63)


def image_suffix(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


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
    seed = request_seed(request.seed)
    prefix = f"nvfp4-api/{uuid.uuid4().hex}"

    async with generation_lock:
        try:
            encoded = await execute_workflow(workflow(prompt, width, height, seed, prefix))
        except (httpx.HTTPError, KeyError, RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"created": int(time.time()), "data": [{"b64_json": encoded}]}


@app.post("/v1/images/edits")
async def edit(
    image: list[UploadFile] = File(...),
    prompt: str = Form(...),
    model: str = Form(MODEL_ID),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    seed: int | None = Form(None),
    response_format: str = Form("b64_json"),
) -> dict[str, Any]:
    if model not in MODEL_ALIASES:
        raise HTTPException(status_code=400, detail=f"model mismatch: {model}")
    if n != 1:
        raise HTTPException(status_code=400, detail="only n=1 is supported")
    if response_format != "b64_json":
        raise HTTPException(status_code=400, detail="only b64_json is supported")
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if not 1 <= len(image) <= 4:
        raise HTTPException(status_code=400, detail="between 1 and 4 reference images are required")
    width, height = parse_size(size)
    upload_dir = INPUT_ROOT / "nvfp4-api"
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    reference_names: list[str] = []
    try:
        for upload in image:
            data = await upload.read(40 << 20)
            if not data or len(data) >= 40 << 20:
                raise HTTPException(status_code=400, detail="each reference image must be smaller than 40 MiB")
            suffix = image_suffix(data)
            if suffix is None:
                raise HTTPException(status_code=400, detail="references must be PNG, JPEG, or WebP images")
            path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
            path.write_bytes(data)
            paths.append(path)
            reference_names.append(f"nvfp4-api/{path.name}")

        prefix = f"nvfp4-api/{uuid.uuid4().hex}"
        async with generation_lock:
            try:
                encoded = await execute_workflow(
                    workflow(prompt, width, height, request_seed(seed), prefix, reference_names)
                )
            except (httpx.HTTPError, KeyError, RuntimeError, TimeoutError) as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for path in paths:
            path.unlink(missing_ok=True)

    return {"created": int(time.time()), "data": [{"b64_json": encoded}]}
