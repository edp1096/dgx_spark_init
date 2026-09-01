#!/usr/bin/env python3
"""Small image/video facade for the SeedVR2 ComfyUI nodes."""

from __future__ import annotations

import asyncio
import aiohttp
import base64
import binascii
import io
import json
import secrets
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
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
runtime_loaded = False
runtime_preparing = False
runtime_started_at = 0.0
runtime_last_load_seconds = 0.0
runtime_error = ""
runtime_operation: dict[str, Any] | None = None
runtime_operation_history: list[dict[str, Any]] = []


class UpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image: str
    model: str = MODEL_ID
    scale: int = 2
    seed: int | None = None
    response_format: str = "b64_json"
    output_format: str = "png"
    operation_id: str = ""


class RuntimePrepareRequest(BaseModel):
    operation_id: str = ""


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
                "swap_io_components": False, "offload_device": "cpu",
                "cache_model": True, "attention_mode": "sdpa",
            },
        },
        "3": {
            "class_type": "SeedVR2LoadVAEModel",
            "inputs": {
                "model": VAE_MODEL, "device": "cuda:0",
                "encode_tiled": True, "encode_tile_size": 1024, "encode_tile_overlap": 128,
                "decode_tiled": True, "decode_tile_size": 1024, "decode_tile_overlap": 128,
                "tile_debug": "false", "offload_device": "cpu", "cache_model": True,
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
                "swap_io_components": False, "offload_device": "cpu",
                "cache_model": True, "attention_mode": "sdpa",
            },
        },
        "4": {
            "class_type": "SeedVR2LoadVAEModel",
            "inputs": {
                "model": VAE_MODEL, "device": "cuda:0",
                "encode_tiled": True, "encode_tile_size": 1024, "encode_tile_overlap": 128,
                "decode_tiled": True, "decode_tile_size": 1024, "decode_tile_overlap": 128,
                "tile_debug": "false", "offload_device": "cpu", "cache_model": True,
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


def publish_runtime_operation(
    operation_id: str,
    phase: str,
    component: str,
    detail: str,
    progress: float,
    memory_action: str = "",
    resident_after: bool | None = None,
) -> None:
    global runtime_operation, runtime_operation_history
    if not operation_id:
        return
    now = datetime.now(timezone.utc).isoformat()
    started_at = now
    if runtime_operation and runtime_operation.get("operation_id") == operation_id:
        if runtime_operation.get("phase") == phase and runtime_operation.get("component") == component:
            started_at = str(runtime_operation.get("started_at") or now)
    if runtime_operation and runtime_operation.get("operation_id") != operation_id:
        runtime_operation_history = []
    runtime_operation = {
        "operation_id": operation_id,
        "phase": phase,
        "component": component,
        "detail": detail,
        "progress": max(0.0, min(1.0, progress)),
        "memory_action": memory_action,
        "resident_after": resident_after,
        "started_at": started_at,
        "updated_at": now,
    }
    if not runtime_operation_history or any(
        runtime_operation_history[-1].get(key) != runtime_operation.get(key)
        for key in ("phase", "component", "detail", "memory_action", "resident_after")
    ):
        runtime_operation_history.append(dict(runtime_operation))
        runtime_operation_history = runtime_operation_history[-32:]


def seedvr_node_phase(graph: dict[str, Any], node_id: str) -> tuple[str, str, str, float, str]:
    class_type = str((graph.get(str(node_id)) or {}).get("class_type", ""))
    if class_type in {"SeedVR2LoadDiTModel", "SeedVR2LoadVAEModel"}:
        return "model_loading", class_type, "SeedVR2 가중치 탑재·GPU 이동", 0.16, "load"
    if class_type == "SeedVR2VideoUpscaler":
        return "sampling", "SeedVR2 DiT·VAE", "업스케일 추론·VAE 디코딩", 0.5, ""
    if class_type in {"CreateVideo", "SaveVideo", "SaveImage"}:
        return "finalizing", class_type, "업스케일 결과 조립·저장", 0.94, ""
    return "preparing", class_type or "SeedVR2 입력", "업스케일 입력 준비", 0.08, ""


def publish_seed_cache_transition(operation_id: str) -> None:
    # cache_model retains CPU-side weights, while offload_device=cpu releases
    # the active GPU working set. Keep both lifecycle facts visible.
    publish_runtime_operation(
        operation_id, "model_unloading", "SeedVR2 DiT·VAE",
        "GPU 작업 가중치 해제·CPU로 오프로딩", 0.96, "unload", False,
    )
    publish_runtime_operation(
        operation_id, "cache_retaining", "SeedVR2 DiT·VAE",
        "CPU 모델 캐시 유지", 0.98, "retain", True,
    )


async def submit_and_wait(graph: dict[str, Any], operation_id: str) -> tuple[httpx.AsyncClient, dict[str, Any]]:
    # Kept as one websocket-correlated execution so cached loader nodes and the
    # long upscaler node are reported from actual ComfyUI execution events.
    client_id = uuid.uuid4().hex
    ws_url = COMFY_URL.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={client_id}"
    client = httpx.AsyncClient(timeout=60 * 60 * 8)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, heartbeat=30, receive_timeout=60 * 60 * 8) as websocket:
                submitted = await client.post(
                    f"{COMFY_URL}/prompt", json={"prompt": graph, "client_id": client_id}
                )
                submitted.raise_for_status()
                body = submitted.json()
                if body.get("node_errors"):
                    raise RuntimeError(f"invalid SeedVR2 workflow: {body['node_errors']}")
                prompt_id = body["prompt_id"]
                while True:
                    message = await websocket.receive()
                    if message.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(f"SeedVR2 websocket failed: {websocket.exception()}")
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(message.data)
                    data = event.get("data") or {}
                    if data.get("prompt_id") != prompt_id:
                        continue
                    if event.get("type") == "execution_error":
                        raise RuntimeError(f"ComfyUI upscale failed: {data}")
                    if event.get("type") == "executing":
                        node_id = data.get("node")
                        if node_id is None:
                            break
                        phase, component, detail, progress, action = seedvr_node_phase(graph, str(node_id))
                        publish_runtime_operation(operation_id, phase, component, detail, progress, action)
                return client, await wait_for_output(client, prompt_id)
    except Exception:
        await client.aclose()
        raise


async def execute_workflow(graph: dict[str, Any], operation_id: str = "") -> str:
    output: dict[str, Any] | None = None
    client: httpx.AsyncClient | None = None
    try:
        client, output = await submit_and_wait(graph, operation_id)
        viewed = await client.get(f"{COMFY_URL}/view", params=output)
        viewed.raise_for_status()
        return base64.b64encode(viewed.content).decode("ascii")
    finally:
        if client is not None:
            await client.aclose()
        if output is not None:
            candidate = (OUTPUT_ROOT / output["subfolder"] / output["filename"]).resolve()
            if candidate.is_relative_to(OUTPUT_ROOT):
                candidate.unlink(missing_ok=True)


async def execute_video_workflow(graph: dict[str, Any], operation_id: str = "") -> Path:
    client, output = await submit_and_wait(graph, operation_id)
    await client.aclose()
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


def runtime_status() -> dict[str, Any]:
    elapsed = max(0.0, time.monotonic() - runtime_started_at) if runtime_started_at else 0.0
    return {
        "status": "preparing" if runtime_preparing else "ready" if runtime_loaded else "idle",
        "profile": "seedvr2-3b-fp8",
        "loaded": runtime_loaded,
        "preparing": runtime_preparing,
        "elapsed_seconds": round(elapsed, 3) if runtime_preparing else 0,
        "last_load_seconds": round(runtime_last_load_seconds, 3),
        "error": runtime_error,
        "operation": runtime_operation,
        "operation_history": runtime_operation_history,
    }


@app.get("/v1/models/runtime/status")
async def model_runtime_status(operation_id: str = "") -> dict[str, Any]:
    if not await comfy_ready():
        raise HTTPException(status_code=503, detail="SeedVR2 runtime is starting")
    status = runtime_status()
    if operation_id and (not runtime_operation or runtime_operation.get("operation_id") != operation_id):
        status["operation"] = None
        status["operation_history"] = []
    return status


@app.post("/v1/models/runtime/prepare")
async def prepare_runtime(request: RuntimePrepareRequest) -> dict[str, Any]:
    global runtime_loaded, runtime_preparing, runtime_started_at
    global runtime_last_load_seconds, runtime_error
    if runtime_loaded:
        return {**runtime_status(), "prepared": True, "warm": True}
    if upscale_lock.locked():
        raise HTTPException(status_code=409, detail="another SeedVR2 operation is running")
    runtime_preparing = True
    runtime_started_at = time.monotonic()
    runtime_error = ""
    publish_runtime_operation(request.operation_id, "preparing", MODEL_ID, "SeedVR2 준비 입력 생성", 0.03)
    image_name = ""
    input_path: Path | None = None
    try:
        image_name, input_path = save_input(Image.new("RGB", (256, 256), (127, 127, 127)))
        prefix = f"seedvr2-prepare/{uuid.uuid4().hex}"
        async with upscale_lock:
            publish_runtime_operation(
                request.operation_id, "model_loading", "SeedVR2 DiT·VAE", "가중치 탑재·GPU 이동", 0.12, "load"
            )
            await execute_workflow(workflow(image_name, 512, 0, prefix), request.operation_id)
        runtime_loaded = True
        runtime_last_load_seconds = time.monotonic() - runtime_started_at
        publish_seed_cache_transition(request.operation_id)
        publish_runtime_operation(
            request.operation_id, "completed", MODEL_ID, "SeedVR2 런타임 준비 완료", 1.0, "retain", True
        )
        # Build the response after clearing the transient flag. A return value
        # is evaluated before ``finally`` runs, which otherwise reports
        # status=preparing even though the synchronous warmup has completed.
        runtime_preparing = False
        return {**runtime_status(), "prepared": True, "warm": False}
    except Exception as exc:
        runtime_error = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        runtime_preparing = False
        if input_path is not None:
            input_path.unlink(missing_ok=True)


@app.get("/health")
async def health() -> dict[str, Any]:
    if not await comfy_ready():
        raise HTTPException(status_code=503, detail="SeedVR2 runtime is starting")
    return {"status": "ok", "busy": upscale_lock.locked(), "runtime": runtime_status()}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


@app.post("/v1/images/upscale")
async def upscale(request: UpscaleRequest) -> dict[str, Any]:
    global runtime_loaded
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
        publish_runtime_operation(request.operation_id, "preparing", MODEL_ID, "업스케일 입력 준비", 0.04)
        async with upscale_lock:
            encoded = await execute_workflow(
                workflow(image_name, min(target_width, target_height), seed, prefix), request.operation_id
            )
        runtime_loaded = True
        publish_seed_cache_transition(request.operation_id)
        publish_runtime_operation(
            request.operation_id, "completed", MODEL_ID, "이미지 업스케일 완료", 1.0, "retain", True
        )
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
    operation_id: str = Form(""),
) -> FileResponse:
    global runtime_loaded
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
        publish_runtime_operation(operation_id, "preparing", MODEL_ID, "영상 입력·구간 준비", 0.03)
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
                video_workflow(workflow_relative, short_edge, actual_seed, prefix, batch_size, temporal_overlap),
                operation_id,
            )
        runtime_loaded = True
        publish_seed_cache_transition(operation_id)
        publish_runtime_operation(
            operation_id, "completed", MODEL_ID, "영상 업스케일 완료", 1.0, "retain", True
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
