#!/usr/bin/env python3
"""Small image/video facade for the SeedVR2 ComfyUI nodes."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import secrets
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
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


app = FastAPI(title="SeedVR2 Upscale API")


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


def video_workflow(video_name: str, short_edge: int, seed: int, prefix: str,
                   batch_size: int, temporal_overlap: int) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {
            "class_type": "SeedVR2LoadDiTModel",
            "inputs": {
                "model": DIT_MODEL, "device": "cuda:0", "blocks_to_swap": 0,
                "swap_io_components": False, "offload_device": "none",
                "cache_model": False, "attention_mode": "sdpa",
            },
        },
        "4": {
            "class_type": "SeedVR2LoadVAEModel",
            "inputs": {
                "model": VAE_MODEL, "device": "cuda:0",
                "encode_tiled": True, "encode_tile_size": 1024, "encode_tile_overlap": 128,
                "decode_tiled": True, "decode_tile_size": 1024, "decode_tile_overlap": 128,
                "tile_debug": "false", "offload_device": "none", "cache_model": False,
            },
        },
        "5": {
            "class_type": "SeedVR2VideoUpscaler",
            "inputs": {
                "image": ["2", 0], "dit": ["3", 0], "vae": ["4", 0], "seed": seed,
                "resolution": short_edge, "max_resolution": 4096, "batch_size": batch_size,
                "uniform_batch_size": True, "temporal_overlap": temporal_overlap,
                "prepend_frames": 0, "color_correction": "lab", "input_noise_scale": 0.0,
                "latent_noise_scale": 0.0, "offload_device": "cpu", "enable_debug": False,
            },
        },
        "6": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["5", 0], "fps": ["2", 2], "audio": ["2", 1], "bit_depth": ["2", 3]},
        },
        "7": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["6", 0], "filename_prefix": prefix, "format": "mp4", "codec": "auto"},
        },
    }


async def comfy_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"{COMFY_URL}/object_info")
        return response.is_success
    except httpx.HTTPError:
        return False


@app.post("/v1/cancel")
async def cancel_generation() -> dict[str, str]:
    """Interrupt the single active ComfyUI upscale, if any."""
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.post(f"{COMFY_URL}/interrupt")
        response.raise_for_status()
    return {"status": "cancelling"}


async def wait_for_output(client: httpx.AsyncClient, prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 60 * 60 * 8
    while time.monotonic() < deadline:
        response = await client.get(f"{COMFY_URL}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json().get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI upscale failed: {status.get('messages', [])}")
            for output in history.get("outputs", {}).values():
                for key in ("images", "videos", "audio"):
                    assets = output.get(key, [])
                    if assets:
                        return assets[0]
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


async def execute_video_workflow(graph: dict[str, Any]) -> Path:
    async with httpx.AsyncClient(timeout=60 * 60 * 8) as client:
        submitted = await client.post(f"{COMFY_URL}/prompt", json={"prompt": graph})
        submitted.raise_for_status()
        body = submitted.json()
        if body.get("node_errors"):
            raise RuntimeError(f"invalid SeedVR2 video workflow: {body['node_errors']}")
        output = await wait_for_output(client, body["prompt_id"])
    candidate = (OUTPUT_ROOT / output.get("subfolder", "") / output["filename"]).resolve()
    if not candidate.is_relative_to(OUTPUT_ROOT) or not candidate.is_file():
        raise RuntimeError("SeedVR2 did not produce a video file")
    return candidate


def probe_video(path: Path) -> tuple[int, int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="video metadata could not be read")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        duration = frames / fps if frames > 0 and fps > 0 else 0
        if width <= 0 or height <= 0 or duration <= 0:
            raise HTTPException(status_code=400, detail="video metadata could not be read")
        return width, height, duration
    finally:
        capture.release()


@app.get("/health")
async def health() -> dict[str, Any]:
    if not await comfy_ready():
        raise HTTPException(status_code=503, detail="SeedVR2 runtime is starting")
    return {"status": "ok", "busy": upscale_lock.locked()}


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


@app.post("/v1/videos/upscale")
async def upscale_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    scale: float = Form(2),
    seed: int = Form(-1),
    batch_size: int = Form(5),
    temporal_overlap: int = Form(1),
    start_time: float = Form(0),
    end_time: float = Form(0),
) -> FileResponse:
    if scale <= 1 or scale > 4:
        raise HTTPException(status_code=400, detail="scale must be greater than 1 and at most 4")
    if batch_size < 1 or (batch_size - 1) % 4 != 0 or batch_size > 21:
        raise HTTPException(status_code=400, detail="batch_size must be one of 1, 5, 9, 13, 17 or 21")
    if temporal_overlap < 0 or temporal_overlap > 4:
        raise HTTPException(status_code=400, detail="temporal_overlap must be between 0 and 4")
    suffix = Path(video.filename or "source.mp4").suffix.lower()
    if suffix not in {".mp4", ".webm", ".mov", ".mkv", ".m4v"}:
        suffix = ".mp4"
    relative = f"seedvr2-api/{uuid.uuid4().hex}{suffix}"
    input_path = INPUT_ROOT / relative
    workflow_path = input_path
    workflow_relative = relative
    input_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with input_path.open("wb") as destination:
            shutil.copyfileobj(video.file, destination, length=8 << 20)
        width, height, duration = probe_video(input_path)
        if duration <= 0:
            raise HTTPException(status_code=400, detail="video has no readable duration")
        short_edge = round(min(width, height) * scale)
        if max(width, height) * scale > 4096:
            raise HTTPException(status_code=400, detail="upscaled video must not exceed 4096 pixels on either edge")
        if start_time < 0 or end_time < 0 or start_time >= duration:
            raise HTTPException(status_code=400, detail="invalid video trim range")
        if duration > 60 and end_time <= 0:
            raise HTTPException(status_code=400, detail="videos longer than 60 seconds require a trim range")
        if end_time > 0:
            if end_time <= start_time or end_time > duration + 0.1:
                raise HTTPException(status_code=400, detail="invalid video trim range")
            if end_time - start_time > 60.1:
                raise HTTPException(status_code=400, detail="video upscale range must not exceed 60 seconds")
            workflow_relative = f"seedvr2-api/{uuid.uuid4().hex}-trimmed.mp4"
            workflow_path = INPUT_ROOT / workflow_relative
            trim_duration = end_time - start_time
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(start_time), "-i", str(input_path), "-t", str(trim_duration),
                "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                str(workflow_path),
            ]
            try:
                subprocess.run(command, check=True, timeout=max(120, int(trim_duration * 4)))
            except (subprocess.SubprocessError, OSError) as exc:
                raise HTTPException(status_code=500, detail="video trim failed") from exc
        actual_seed = seed if seed >= 0 else secrets.randbits(32)
        prefix = f"seedvr2-api/{uuid.uuid4().hex}"
        async with upscale_lock:
            output_path = await execute_video_workflow(
                video_workflow(workflow_relative, short_edge, actual_seed, prefix, batch_size, temporal_overlap)
            )
        background_tasks.add_task(output_path.unlink, missing_ok=True)
        return FileResponse(
            output_path, media_type="video/mp4", filename=f"{Path(video.filename or 'video').stem}-upscaled.mp4",
            headers={
                "X-SeedVR2-Seed": str(actual_seed),
                "X-SeedVR2-Scale": str(scale),
                "X-SeedVR2-Width": str(round(width * scale)),
                "X-SeedVR2-Height": str(round(height * scale)),
            },
        )
    finally:
        input_path.unlink(missing_ok=True)
        if workflow_path != input_path:
            workflow_path.unlink(missing_ok=True)
