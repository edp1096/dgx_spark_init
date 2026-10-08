#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import io
import time
import uuid
from pathlib import Path
from typing import Any, Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from PIL import Image, ImageStat, UnidentifiedImageError


COMFY_URL = "http://127.0.0.1:8190"
INPUT_ROOT = Path("/opt/ComfyUI/input").resolve()
OUTPUT_ROOT = Path("/opt/ComfyUI/output").resolve()
MODEL_ROOT = Path("/opt/ComfyUI/models")
SWAP_MODEL = "inswapper_128.onnx"
MAX_IMAGE_BYTES = 64 << 20
swap_lock = asyncio.Lock()

app = FastAPI(title="Spark Media ReActor Face Swap", version="1.0.0")


async def read_image(upload: UploadFile, role: str) -> Image.Image:
    data = await upload.read(MAX_IMAGE_BYTES + 1)
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(400, f"{role} image must be 1 byte..64 MiB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, f"{role} image is invalid") from error


def save_input(image: Image.Image, role: str) -> tuple[str, Path]:
    relative = f"reactor-api/{uuid.uuid4().hex}-{role}.png"
    path = INPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return relative, path


def workflow(target_name: str, source_name: str, target_index: int, source_index: int, prefix: str) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": target_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        "3": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["1", 0],
                "source_image": ["2", 0],
                "swap_model": SWAP_MODEL,
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "none",
                "face_restore_visibility": 1.0,
                "codeformer_weight": 0.5,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": str(target_index),
                "source_faces_index": str(source_index),
                "console_log_level": 1,
            },
        },
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["3", 0]}},
    }


async def wait_ready(timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=3) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{COMFY_URL}/object_info/ReActorFaceSwap")
                if response.is_success and "ReActorFaceSwap" in response.json():
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise HTTPException(503, "ReActor ComfyUI node is not ready")


async def run_workflow(graph: dict[str, Any]) -> bytes:
    await wait_ready()
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(f"{COMFY_URL}/prompt", json={"prompt": graph})
        response.raise_for_status()
        body = response.json()
        if body.get("node_errors"):
            raise RuntimeError(f"invalid ReActor workflow: {body['node_errors']}")
        prompt_id = body["prompt_id"]
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            history_response = await client.get(f"{COMFY_URL}/history/{prompt_id}")
            history_response.raise_for_status()
            history = history_response.json().get(prompt_id)
            if history:
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ReActor failed: {status.get('messages', [])}")
                for output in history.get("outputs", {}).values():
                    images = output.get("images") or []
                    if images:
                        asset = images[0]
                        params = {key: asset.get(key, "") for key in ("filename", "subfolder", "type")}
                        result = await client.get(f"{COMFY_URL}/view", params=params)
                        result.raise_for_status()
                        return result.content
            await asyncio.sleep(0.25)
    raise TimeoutError("ReActor face swap timed out")


@app.get("/health")
async def health(response: Response) -> dict[str, Any]:
    ready = False
    providers: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"{COMFY_URL}/object_info/ReActorFaceSwap")
            ready = response.is_success and "ReActorFaceSwap" in response.json()
    except httpx.HTTPError:
        pass
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
    except Exception:
        pass
    if not ready:
        response.status_code = 503
    return {
        "status": "ok" if ready else "starting",
        "ready": ready,
        "model": SWAP_MODEL,
        "providers": providers,
        "models_ready": (MODEL_ROOT / "insightface" / SWAP_MODEL).is_file(),
    }


@app.post("/v1/faces/swap")
async def swap_faces(
    target: Annotated[UploadFile, File()],
    source: Annotated[UploadFile, File()],
    target_face_index: Annotated[int, Form()] = 0,
    source_face_index: Annotated[int, Form()] = 0,
    operation_id: Annotated[str, Form()] = "",
) -> dict[str, Any]:
    if target_face_index < 0 or target_face_index > 15 or source_face_index < 0 or source_face_index > 15:
        raise HTTPException(400, "face indexes must be between 0 and 15")
    target_image = await read_image(target, "target")
    source_image = await read_image(source, "source")
    target_name, target_path = save_input(target_image, "target")
    source_name, source_path = save_input(source_image, "source")
    prefix = f"reactor-api/{operation_id or uuid.uuid4().hex}"
    try:
        async with swap_lock:
            result = await run_workflow(workflow(target_name, source_name, target_face_index, source_face_index, prefix))
        output = Image.open(io.BytesIO(result)).convert("RGB")
        extrema = ImageStat.Stat(output).extrema
        if output.size == (512, 512) and all(high <= 1 for _low, high in extrema):
            raise HTTPException(422, "ReActor rejected the target image or could not produce a face swap")
        return {
            "data": [{"b64_json": base64.b64encode(result).decode("ascii")}],
            "model": SWAP_MODEL,
            "target_face_index": target_face_index,
            "source_face_index": source_face_index,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(422, str(error)) from error
    finally:
        target_path.unlink(missing_ok=True)
        source_path.unlink(missing_ok=True)


@app.post("/v1/cancel")
async def cancel() -> dict[str, str]:
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.post(f"{COMFY_URL}/interrupt")
        response.raise_for_status()
    return {"status": "cancelling"}
