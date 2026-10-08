from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile


COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8190").rstrip("/")
INPUT_DIR = Path("/opt/ComfyUI/input")
OUTPUT_DIR = Path("/opt/ComfyUI/output")
DIRECTIONS = ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left")
FRAME_INDICES = (0, 31, 46, 62, 77, 93, 109, 116)
TURN_LOCK = asyncio.Lock()
OPERATIONS: dict[str, dict] = {}

app = FastAPI(title="MiniMax H3 Character Turntable")


def graph(image_name: str, seed: int, prefix: str) -> dict:
    prompt = """<Picture 1> is the only identity and appearance reference.
Create one continuous five-second studio turntable shot of exactly the same visible subject. The subject remains perfectly still in a neutral relaxed pose at the exact center of a plain warm light-grey studio. Preserve the exact identity, anatomy and proportions, face or head design, skin/fur/materials, hair, clothing, footwear, accessories, colors, textures, and every distinctive visible feature throughout the shot.

The camera performs one smooth clockwise 360-degree orbit at eye level, beginning at a complete front view, passing through front-right three-quarter, strict right profile, back-right three-quarter, complete back view, back-left three-quarter, strict left profile, front-left three-quarter, and ending at the front. Keep camera distance, focal length, framing, horizon, exposure, lighting, background, subject scale, and ground position locked. Keep the complete subject visible.

Single unbroken shot. No cuts, zoom, camera height change, subject motion, expression change, anatomy change, design change, wardrobe change, identity drift, extra subject, extra limb, text, label, watermark, or sound."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength_model": 1.0}},
        "3": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["2", 0], "attention": "comfy kitchen attention"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt, "width": 768, "height": 1344, "length": 124, "ref_image_size": "max", "ref_images.ref_image_0": ["7", 0]}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "BasicScheduler", "inputs": {"model": ["3", 0], "scheduler": "simple", "steps": 4, "denoise": 1.0}},
        "12": {"class_type": "BasicGuider", "inputs": {"model": ["3", 0], "conditioning": ["8", 0]}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9", 0], "guider": ["12", 0], "sampler": ["10", 0], "sigmas": ["11", 0], "latent_image": ["8", 1]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["5", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "fps": 24.0, "bit_depth": 8}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": prefix, "format": "mp4", "codec": "h264"}},
    }


def set_operation(operation_id: str, phase: str, detail: str, progress: float) -> None:
    OPERATIONS[operation_id] = {"operation_id": operation_id, "phase": phase, "detail": detail, "progress": progress, "updated_at": time.time()}


def is_image_data(data: bytes) -> bool:
    return (
        data.startswith(b"\x89PNG\r\n\x1a\n")
        or data.startswith(b"\xff\xd8\xff")
        or (len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )


async def wait_for_comfy() -> None:
    async with httpx.AsyncClient(timeout=3) as client:
        for _ in range(120):
            try:
                if (await client.get(f"{COMFY_URL}/system_stats")).is_success:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise HTTPException(503, "MiniMax H3 engine is not ready")


def extract_frames(video: Path, target: Path) -> list[dict]:
    target.mkdir(parents=True, exist_ok=True)
    result = []
    for direction, frame_index in zip(DIRECTIONS, FRAME_INDICES):
        destination = target / f"{direction}.jpg"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-vf", f"select=eq(n\\,{frame_index})", "-frames:v", "1", "-q:v", "2", str(destination),
        ], check=True, timeout=180)
        data = destination.read_bytes()
        result.append({"direction": direction, "frame_index": frame_index, "mime_type": "image/jpeg", "data": base64.b64encode(data).decode("ascii")})
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "minimax-h3"}


@app.get("/v1/character/turntable/status")
async def turntable_status(operation_id: str):
    return {"operation": OPERATIONS.get(operation_id, {"operation_id": operation_id, "phase": "waiting", "detail": "작업 대기", "progress": 0.02})}


@app.post("/v1/character/turntable")
async def create_turntable(image: UploadFile = File(...), operation_id: str = Form(...), seed: int = Form(-1)):
    if not operation_id.startswith("character-turntable-") or len(operation_id) > 128:
        raise HTTPException(400, "invalid operation id")
    set_operation(operation_id, "queued", "MMH3 작업 순서 확인", 0.02)
    async with TURN_LOCK:
        await wait_for_comfy()
        token = uuid.uuid4().hex
        suffix = Path(image.filename or "reference.png").suffix.lower() or ".png"
        input_name = f"character-turntable-{token}{suffix}"
        input_path = INPUT_DIR / input_name
        image_data = await image.read()
        if not image_data:
            input_path.unlink(missing_ok=True)
            set_operation(operation_id, "failed", "이미지가 비어 있습니다.", 1.0)
            raise HTTPException(400, "image is empty")
        if not is_image_data(image_data):
            set_operation(operation_id, "failed", "지원하는 이미지 파일이 아닙니다.", 1.0)
            raise HTTPException(400, "uploaded data is not a PNG, JPEG, or WebP image")
        input_path.write_bytes(image_data)
        if seed < 0:
            seed = int.from_bytes(os.urandom(8), "big") & 0x7fffffffffffffff
        prefix = f"character-turntable/{token}"
        set_operation(operation_id, "preparing", "MMH3 모델·참조 이미지 준비", 0.05)
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                queued = await client.post(f"{COMFY_URL}/prompt", json={"prompt": graph(input_name, seed, prefix), "client_id": operation_id})
                queued.raise_for_status()
                prompt_id = queued.json()["prompt_id"]
                started = time.monotonic()
                while True:
                    history_response = await client.get(f"{COMFY_URL}/history/{prompt_id}")
                    history_response.raise_for_status()
                    item = history_response.json().get(prompt_id)
                    elapsed = time.monotonic() - started
                    set_operation(operation_id, "sampling", "360° 회전 영상 생성", min(0.88, 0.10 + elapsed / 760.0))
                    if item:
                        status = item.get("status", {})
                        if status.get("status_str") == "error":
                            raise RuntimeError(json.dumps(status, ensure_ascii=False))
                        outputs = item.get("outputs", {}).get("16", {})
                        # SaveVideo is represented as an animated item. ComfyUI
                        # versions expose it under images, videos, or gifs.
                        videos = outputs.get("videos") or outputs.get("gifs") or outputs.get("images") or []
                        if videos:
                            output = videos[0]
                            video_path = OUTPUT_DIR / output.get("subfolder", "") / output["filename"]
                            break
                    await asyncio.sleep(2)
            set_operation(operation_id, "extracting", "8방향 프레임 추출", 0.92)
            with tempfile.TemporaryDirectory(prefix="h3-frames-") as temp:
                frames = await asyncio.to_thread(extract_frames, video_path, Path(temp))
            set_operation(operation_id, "completed", "8방향 준비 완료", 1.0)
            return {"operation_id": operation_id, "seed": seed, "frames": frames}
        except Exception as exc:
            set_operation(operation_id, "failed", str(exc), 1.0)
            raise HTTPException(502, f"MMH3 turntable failed: {exc}") from exc
        finally:
            await image.close()
            input_path.unlink(missing_ok=True)
