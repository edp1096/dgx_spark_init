#!/usr/bin/env python3
"""OpenAI-compatible facade for Krea 2 Turbo NVFP4 on ComfyUI."""

from __future__ import annotations

import asyncio
import aiohttp
import base64
import binascii
import ctypes
import gc
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from civitai_models import (
    BF16_SOURCES,
    CHECKPOINTS,
    HERETIC_TEXT_ENCODER,
    NVFP4_FILENAMES,
    download_checkpoint,
    link_model,
    link_nvfp4,
    link_text_encoder,
    model_path,
    nvfp4_path,
    nvfp4_ready,
    nvfp4_validated,
    nvfp4_validation_path,
    ready,
    save_token,
    stored_token,
    token_configured,
)
from krea2_nvfp4 import PROFILE_COMMIT, PROFILE_SOURCE, convert_krea2_nvfp4


COMFY_URL = "http://127.0.0.1:8188"
MODEL_ID = "krea2-turbo-nvfp4"
MODEL_ALIASES = {MODEL_ID, "krea/Krea-2-Turbo"}
DIFFUSION_MODEL = "krea2_turbo_nvfp4.safetensors"
OFFICIAL_INT8_MODEL = "krea2_turbo_int8_convrot.safetensors"
CHECKPOINT_MODELS = {
    "official": DIFFUSION_MODEL,
    "official-int8": OFFICIAL_INT8_MODEL,
    **{key: checkpoint.filename for key, checkpoint in CHECKPOINTS.items()},
    **{f"{key}-nvfp4": filename for key, filename in NVFP4_FILENAMES.items()},
}
CHECKPOINT_SAMPLING = {
    "moody-v7": ("euler_ancestral", "beta"),
    "moody-cutie-v4": ("euler_ancestral", "beta"),
    "moody-amateur-v1": ("euler_ancestral", "beta"),
}
OFFICIAL_CHECKPOINTS = {"official", "official-int8"}
STYLE_REFERENCE_MODEL = OFFICIAL_INT8_MODEL
HEAD_SWAP_MODEL = OFFICIAL_INT8_MODEL
TEXT_ENCODER = "qwen3vl_4b_fp8_scaled.safetensors"
VISION_TEXT_ENCODER = "qwen3vl_4b_bf16.safetensors"
VISION_INSTRUCT_SYSTEM = (
    "Describe the key features of the reference image (color, shape, size, texture, objects, "
    "background), then explain how the user's instruction should combine with or alter it, and "
    "generate a new image meeting the instruction while staying consistent with the reference "
    "where appropriate:"
)
VAE = "qwen_image_vae.safetensors"
REAL_VAE = "krea2RealVae_v10.safetensors"
WAN_VAE = "wan_2.1_vae.safetensors"
DEPTH_CONTROL_LORA = "krea2-depth-control-lora.safetensors"
IDENTITY_EDIT_LORA = "krea2_identity_edit_v1_2.safetensors"
HEAD_SWAP_LORA = "bfs_head_swap_v1.1_krea2.safetensors"
REID_LORA = "krea2_reid_rank32.safetensors"
CHARACTER_SHEET_LORA = "QuadView_krea2_v1.safetensors"
IDENTITY_EDIT_MODEL = "Krea2_Turbo_convrot_int8mixed.safetensors"
IDENTITY_EDIT_TEXT_ENCODER = "qwen3VLInstruct4bHeretic_int8Convrot.safetensors"
FILTER_BYPASS_BALANCED = "fedor_bypass.safetensors"
FILTER_BYPASS_STRONG = "krea2filterbypass3.safetensors"
FILTER_BYPASS_ADHERENCE = "user/skc3vo.safetensors"
DETAIL_ENHANCER_LORA = "krea-detail-enhancer-exp.safetensors"
NK2E_EDIT_LORA = "NK2E-v0.3.safetensors"
NK2E_CANNY_LORA = "NK2E-canny-v0.1.safetensors"
ANYPAINT_LORA = "krea2_anypaint_rank32.safetensors"
STYLE_REFERENCE_LORA = "krea2_style_reference.safetensors"
STYLE_TRIGGERS = {
    "darkbrush": "monochrome ink wash style",
    "dotmatrix": "monochrome stippling style",
    "kidsdrawing": "naive expressive sketch style",
    "neondrip": "textured abstract style",
    "rainywindow": "rainy window style",
    "retroanime": "purple retro anime style",
    "softwatercolor": "art deco watercolor style",
    "sunsetblur": "ethereal motion blur style",
    "vintagetarot": "vintage tarot style",
}
STYLE_LORAS = {name: f"krea2_{name}.safetensors" for name in STYLE_TRIGGERS}
USER_LORA_ROOT = (Path("/opt/ComfyUI/models/loras/user")).resolve()
HF_TOKEN_FILE = Path("/root/.cache/huggingface/media-secrets/hf_token")
MAX_USER_LORA_BYTES = 2 * 1024 * 1024 * 1024
MAX_USER_LORA_PREVIEW_BYTES = 20 * 1024 * 1024
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
OUTPUT_ROOT = Path("/opt/ComfyUI/output").resolve()
INPUT_ROOT = Path("/opt/ComfyUI/input").resolve()
generation_lock = asyncio.Lock()
segmentation_lock = asyncio.Lock()
depth_processor: Any | None = None
depth_model: Any | None = None
checkpoint_prepare_lock = asyncio.Lock()
checkpoint_prepare_task: asyncio.Task[None] | None = None
checkpoint_prepare_error = ""
checkpoint_prepare_current = ""
checkpoint_prepare_bytes = 0
checkpoint_prepare_total = 0
checkpoint_conversion_task: asyncio.Task[None] | None = None
checkpoint_conversion_error = ""
checkpoint_conversion_current = ""
checkpoint_conversion_stage = ""
checkpoint_conversion_done = 0
checkpoint_conversion_total = 0
runtime_profile = ""
runtime_signature = ""
runtime_stage = "idle"
runtime_started_at = 0.0
runtime_last_load_seconds = 0.0
runtime_error = ""
runtime_operation: dict[str, Any] | None = None
runtime_operation_history: list[dict[str, Any]] = []


class StyleSelection(BaseModel):
    name: str
    strength: float = 1.0


class UserLoRASelection(BaseModel):
    filename: str
    strength: float = 1.0


class UserLoRAImportRequest(BaseModel):
    source: str = Field(min_length=1, max_length=2048)
    provider: str = "auto"
    name: str = Field(default="", max_length=128)
    trigger_word: str = Field(default="", max_length=512)
    memo: str = Field(default="", max_length=2000)
    base_model: str = Field(default="", max_length=128)
    recommended_strength: float = Field(default=1.0, ge=-2.0, le=2.0)
    civitai_token: str = Field(default="", max_length=512)
    hf_token: str = Field(default="", max_length=512)


class UserLoRAUpdateRequest(BaseModel):
    name: str = Field(default="", max_length=128)
    trigger_word: str = Field(default="", max_length=512)
    memo: str = Field(default="", max_length=2000)
    base_model: str = Field(default="", max_length=128)
    recommended_strength: float = Field(default=1.0, ge=-2.0, le=2.0)


class DownloadCredentialRequest(BaseModel):
    civitai_token: str = Field(default="", max_length=512)
    hf_token: str = Field(default="", max_length=512)


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str
    model: str = MODEL_ID
    checkpoint: str = "official-int8"
    n: int = 1
    size: str = "1024x1024"
    seed: int | None = None
    response_format: str = "b64_json"
    control_image: str | None = None
    control_strength: float = 1.0
    control_prompt: str = ""
    prepare_pose_reference: bool = False
    source_image: str | None = None
    reid_image: str | None = None
    character_sheet_image: str | None = None
    reference_image: str | None = None
    reference_images: list[str] = Field(default_factory=list)
    identity_mask: str | None = None
    strict_mask: str | None = None
    strict_mask_grow: int = 0
    strict_mask_feather: float = 0.0
    vae_mode: str = "default"
    identity_fit_mode: str = "fit"
    identity_model: str = "convrot"
    identity_encoder: str = "heretic"
    identity_preset: str = ""
    identity_strength: float = 1.0
    ref_boost: float = 4.0
    source_ref_boost: float = 1.0
    grounding_px: int = 768
    steps: int | None = None
    sampler_name: str | None = None
    scheduler: str | None = None
    style: str | None = None
    style_strength: float = 1.0
    styles: list[StyleSelection] = Field(default_factory=list)
    user_loras: list[UserLoRASelection] = Field(default_factory=list)
    vision_images: list[str] = Field(default_factory=list)
    vision_mode: str = "descriptor"
    vision_megapixels: float = 1.0
    style_reference_images: list[str] = Field(default_factory=list)
    style_reference_strength: float = 1.0
    nk2e_image: str | None = None
    nk2e_mode: str = "edit"
    nk2e_strength: float = 0.7
    nk2e_preprocessed: bool = False
    anypaint_image: str | None = None
    anypaint_mask: str | None = None
    outpaint_left: int = 0
    outpaint_top: int = 0
    outpaint_right: int = 0
    outpaint_bottom: int = 0
    anypaint_strength: float = 1.0
    anypaint_reference_max_edge: int = 384
    anypaint_boundary_redraw_px: int = 32
    anypaint_vlm_reference: bool = True
    filter_mode: str = "balanced"
    filter_strength: float | None = None
    prompt_enhancer: bool = False
    prompt_enhancer_strength: float = 1.0
    prompt_text_scale: float = 1.75
    detail_enhance_image: str | None = None
    detail_strength: float = 1.0
    detail_vae: str = "wan"
    prepare_only: bool = False
    runtime_profile: str = Field(default="", max_length=128)
    operation_id: str = Field(default="", max_length=128)


class SegmentRequest(BaseModel):
    image: str
    prompt: str
    box_threshold: float = 0.3
    text_threshold: float = 0.2
    mask_threshold: float = 0.5
    grow: int = 8
    feather: float = 4.0


class CheckpointPrepareRequest(BaseModel):
    civitai_token: str = ""
    hf_token: str = ""
    variants: list[str] = Field(default_factory=lambda: list(CHECKPOINTS))


class CheckpointConvertRequest(BaseModel):
    civitai_token: str = ""
    variants: list[str] = Field(default_factory=lambda: list(BF16_SOURCES))
    remove_bf16_sources: bool = False


app = FastAPI(title="Krea 2 Turbo NVFP4 API")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _stored_hf_token() -> str:
    try:
        return HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return os.environ.get("HF_TOKEN", "").strip()


def _save_hf_token(token: str) -> None:
    HF_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    HF_TOKEN_FILE.write_text(token, encoding="utf-8")
    HF_TOKEN_FILE.chmod(0o600)


def _safe_lora_filename(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", Path(value).stem).strip(".-") or "user-lora"
    return f"{stem[:96]}.safetensors"


def _user_lora_path(filename: str) -> Path:
    clean = Path(filename).name
    if clean != filename or not clean.lower().endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="invalid LoRA filename")
    return USER_LORA_ROOT / clean


def _user_lora_preview_path(filename: str) -> Path:
    return _user_lora_path(filename).with_suffix(".preview.webp")


def _civitai_headers(token: str) -> dict[str, str]:
    headers = {"User-Agent": "Spark-Media-LoRA/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _civitai_json(path: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"https://civitai.com/api/v1/{path.lstrip('/')}", headers=_civitai_headers(token)
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except Exception as exc:
        raise ValueError(f"Civitai API 요청 실패: {exc}") from exc


def _civitai_version_id(source: str, token: str) -> str:
    if source.isdigit():
        return source
    parsed = urllib.parse.urlparse(source)
    if parsed.hostname not in {"civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"}:
        raise ValueError("Civitai 주소 또는 모델 버전 ID를 입력하세요")
    query = urllib.parse.parse_qs(parsed.query)
    version = (query.get("modelVersionId") or query.get("modelversionid") or [""])[0]
    if version.isdigit():
        return version
    match = re.search(r"/(?:api/download/models|model-versions)/(\d+)", parsed.path, re.I)
    if match:
        return match.group(1)
    model_match = re.search(r"/models/(\d+)", parsed.path, re.I)
    if not model_match:
        raise ValueError("Civitai 주소에서 모델 ID를 찾지 못했습니다")
    model = _civitai_json(f"models/{model_match.group(1)}", token)
    versions = model.get("modelVersions") or []
    if not versions:
        raise ValueError("다운로드 가능한 모델 버전이 없습니다")
    return str(versions[0]["id"])


def _download_url(url: str, destination: Path, headers: dict[str, str]) -> str:
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=900) as response:
            declared = int(response.headers.get("Content-Length") or 0)
            if declared > MAX_USER_LORA_BYTES:
                raise ValueError("LoRA 파일이 2 GiB 제한을 초과합니다")
            total = 0
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_USER_LORA_BYTES:
                        raise ValueError("LoRA 파일이 2 GiB 제한을 초과합니다")
                    output.write(chunk)
                    digest.update(chunk)
        temporary.replace(destination)
        return digest.hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _import_civitai_lora(request: UserLoRAImportRequest) -> dict[str, Any]:
    token = request.civitai_token.strip() or stored_token()
    if request.civitai_token.strip():
        save_token(token)
    version_id = _civitai_version_id(request.source.strip(), token)
    version = _civitai_json(f"model-versions/{version_id}", token)
    model_type = str((version.get("model") or {}).get("type", "")).lower()
    if model_type and model_type != "lora":
        raise ValueError("선택한 Civitai 파일은 LoRA가 아닙니다")
    files = [
        item for item in version.get("files", [])
        if str(item.get("name", "")).lower().endswith(".safetensors") and item.get("downloadUrl")
    ]
    if not files:
        raise ValueError("이 버전에 safetensors LoRA 파일이 없습니다")
    parsed_source = urllib.parse.urlparse(request.source.strip())
    requested_file_id = (urllib.parse.parse_qs(parsed_source.query).get("fileId") or [""])[0]
    if requested_file_id:
        selected = next((item for item in files if str(item.get("id", "")) == requested_file_id), None)
        if selected is None:
            raise ValueError("Civitai 주소가 지정한 safetensors 파일을 이 버전에서 찾지 못했습니다")
    else:
        selected = next((item for item in files if item.get("primary")), files[0])
    filename = _safe_lora_filename(request.name.strip() or str(selected.get("name", "")))
    destination = USER_LORA_ROOT / filename
    if destination.exists():
        raise FileExistsError(f"{filename}이 이미 등록되어 있습니다")
    digest = _download_url(str(selected["downloadUrl"]), destination, _civitai_headers(token))
    expected = str((selected.get("hashes") or {}).get("SHA256", "")).lower()
    if expected and expected != digest.lower():
        destination.unlink(missing_ok=True)
        raise ValueError("Civitai LoRA 체크섬이 일치하지 않습니다")
    trained_words = [str(word).strip() for word in version.get("trainedWords", []) if str(word).strip()]
    return {
        "filename": filename,
        "name": request.name.strip() or Path(filename).stem,
        "trigger_word": request.trigger_word.strip() or ", ".join(trained_words),
        "memo": request.memo.strip(),
        "recommended_strength": request.recommended_strength,
        "source": request.source.strip(),
        "provider": "civitai",
        "civitai_version_id": version_id,
        "base_model": request.base_model.strip() or version.get("baseModel", ""),
        "sha256": digest,
        "created_at": time.time(),
    }


def _parse_hf_source(source: str) -> tuple[str, str | None, str | None]:
    source = source.strip().rstrip("/")
    if source.startswith("http://") or source.startswith("https://"):
        parsed = urllib.parse.urlparse(source)
        if parsed.hostname not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("Hugging Face 주소 또는 owner/repository를 입력하세요")
        parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("Hugging Face 저장소 주소가 올바르지 않습니다")
        repo_id = "/".join(parts[:2])
        if len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
            return repo_id, "/".join(parts[4:]), parts[3]
        return repo_id, None, None
    parts = source.split("/")
    if len(parts) < 2:
        raise ValueError("Hugging Face 저장소는 owner/repository 형식이어야 합니다")
    return "/".join(parts[:2]), "/".join(parts[2:]) or None, None


def _import_hf_lora(request: UserLoRAImportRequest) -> dict[str, Any]:
    token = request.hf_token.strip() or _stored_hf_token()
    if request.hf_token.strip():
        _save_hf_token(token)
    repo_id, requested_file, revision = _parse_hf_source(request.source)
    api = HfApi(token=token or None)
    files = api.list_repo_files(repo_id=repo_id, revision=revision, token=token or None)
    if requested_file:
        candidates = [item for item in files if item == requested_file]
        if not candidates:
            raise ValueError(f"저장소에서 {requested_file} 파일을 찾지 못했습니다")
    else:
        candidates = [item for item in files if item.lower().endswith(".safetensors")]
        if len(candidates) != 1:
            raise ValueError("safetensors가 여러 개입니다. Hugging Face 파일 페이지 주소를 입력하세요")
    remote_file = candidates[0]
    filename = _safe_lora_filename(request.name.strip() or Path(remote_file).name)
    destination = USER_LORA_ROOT / filename
    if destination.exists():
        raise FileExistsError(f"{filename}이 이미 등록되어 있습니다")
    downloaded = Path(hf_hub_download(repo_id, remote_file, revision=revision, token=token or None))
    if downloaded.stat().st_size > MAX_USER_LORA_BYTES:
        raise ValueError("LoRA 파일이 2 GiB 제한을 초과합니다")
    shutil.copy2(downloaded, destination)
    digest = _file_sha256(destination)
    return {
        "filename": filename,
        "name": request.name.strip() or Path(filename).stem,
        "trigger_word": request.trigger_word.strip(),
        "memo": request.memo.strip(),
        "recommended_strength": request.recommended_strength,
        "source": request.source.strip(),
        "provider": "huggingface",
        "base_model": request.base_model.strip(),
        "repo_id": repo_id,
        "remote_file": remote_file,
        "sha256": digest,
        "created_at": time.time(),
    }


def _list_user_loras() -> list[dict[str, Any]]:
    USER_LORA_ROOT.mkdir(parents=True, exist_ok=True)
    result = []
    for path in sorted(USER_LORA_ROOT.glob("*.safetensors"), key=lambda item: item.name.lower()):
        if not path.is_file() or path.name == "skc3vo.safetensors":
            continue
        metadata = _read_json(path.with_suffix(".json"))
        result.append({
            **metadata,
            "filename": path.name,
            "name": metadata.get("name") or path.stem,
            "trigger_word": metadata.get("trigger_word") or "",
            "memo": metadata.get("memo") or "",
            "recommended_strength": metadata.get("recommended_strength", 1.0),
            "preview_available": path.with_suffix(".preview.webp").is_file(),
            "preview_updated_at": metadata.get("preview_updated_at", 0),
            "size": path.stat().st_size,
        })
    return result


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="size must be WIDTHxHEIGHT") from exc
    if not (512 <= width <= 2048 and 512 <= height <= 2048):
        raise HTTPException(status_code=400, detail="width and height must be between 512 and 2048")
    # Krea 2's published ResolutionSelector workflow uses a multiple of 8
    # (for example 768x1368 at 9:16).  Requiring 16 here silently forced edit
    # reproductions onto a different latent geometry than the reference graph.
    if width % 8 or height % 8:
        raise HTTPException(status_code=400, detail="width and height must be multiples of 8")
    return width, height


def workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    steps: int = 8,
    styles: list[StyleSelection] | None = None,
    user_loras: list[UserLoRASelection] | None = None,
    diffusion_model: str = DIFFUSION_MODEL,
) -> dict[str, Any]:
    graph = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_model, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {
            "class_type": "EmptyLatentImage",
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
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }
    model_input: list[Any] = ["1", 0]
    for offset, style in enumerate(styles or []):
        node_id = str(20 + offset)
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_input,
                "lora_name": STYLE_LORAS[style.name],
                "strength_model": style.strength,
            },
        }
        model_input = [node_id, 0]
    next_id = 20 + len(styles or [])
    for selection in user_loras or []:
        node_id = str(next_id)
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_input,
                "lora_name": f"user/{selection.filename}",
                "strength_model": selection.strength,
            },
        }
        model_input = [node_id, 0]
        next_id += 1
    graph["7"]["inputs"]["model"] = model_input
    return graph


def apply_sampling(graph: dict[str, Any], sampler_name: str, scheduler: str) -> dict[str, Any]:
    """Apply the validated sampling pair to both basic and advanced ComfyUI graphs."""
    for node in graph.values():
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})
        if class_type == "KSampler":
            inputs["sampler_name"] = sampler_name
            inputs["scheduler"] = scheduler
        elif class_type == "KSamplerSelect":
            inputs["sampler_name"] = sampler_name
        elif class_type == "BasicScheduler":
            inputs["scheduler"] = scheduler
    return graph


def apply_filter_bypass(
    graph: dict[str, Any], mode: str, strength: float | None
) -> dict[str, Any]:
    """Apply a selectable filter-relaxation vector before every model adapter."""
    if mode == "off":
        return graph
    lora_name = {
        "adherence": FILTER_BYPASS_ADHERENCE,
        "balanced": FILTER_BYPASS_BALANCED,
        "strong": FILTER_BYPASS_STRONG,
    }[mode]
    default_strength = 0.05 if mode == "adherence" else 1.0
    resolved_strength = strength if strength is not None else default_strength
    patch_id = "900"
    graph[patch_id] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": lora_name,
            "strength_model": resolved_strength,
        },
    }
    for node_id, node in graph.items():
        if node_id == patch_id:
            continue
        inputs = node.get("inputs", {})
        if inputs.get("model") == ["1", 0]:
            inputs["model"] = [patch_id, 0]
    return graph


def apply_prompt_enhancer(
    graph: dict[str, Any], enabled: bool, strength: float, text_scale: float
) -> dict[str, Any]:
    """Wrap each sampler's final model input with the Krea2T prompt enhancer."""
    if not enabled:
        return graph
    next_id = 910
    for node in list(graph.values()):
        if node.get("class_type") not in {"KSampler", "CFGGuider"}:
            continue
        model_input = node.get("inputs", {}).get("model")
        if model_input is None:
            continue
        while str(next_id) in graph:
            next_id += 1
        node_id = str(next_id)
        graph[node_id] = {
            "class_type": "Krea2T-Enhancer-Advanced",
            "inputs": {
                "model": model_input,
                "enabled": True,
                "strength": strength,
                "text_scale": text_scale,
                "debug": False,
            },
        }
        node["inputs"]["model"] = [node_id, 0]
        next_id += 1
    return graph


def apply_vision_conditioning(
    graph: dict[str, Any],
    prompt: str,
    image_names: list[str],
    mode: str,
    vision_megapixels: float,
) -> dict[str, Any]:
    graph["2"]["inputs"]["clip_name"] = VISION_TEXT_ENCODER
    encoder_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "prompt": prompt,
        "vision_megapixels": vision_megapixels,
        "mask_padding": 0.0,
        "vision_position": "before prompt",
        "print_prompt": False,
    }
    for index, image_name in enumerate(image_names, start=1):
        node_id = str(79 + index)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        encoder_inputs[f"image{index}"] = [node_id, 0]
    if mode == "instruct":
        graph["70"] = {
            "class_type": "Krea2SystemPrompt",
            "inputs": {"text": VISION_INSTRUCT_SYSTEM},
        }
        encoder_inputs["system_prompt"] = ["70", 0]
    graph["4"] = {"class_type": "TextEncodeKrea2", "inputs": encoder_inputs}
    return graph


def style_reference_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    image_names: list[str],
    strength: float,
    steps: int,
) -> dict[str, Any]:
    """Official Ostris/ComfyUI Krea 2 style-reference graph."""
    encode_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "prompt": prompt,
        "vae": ["3", 0],
    }
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": STYLE_REFERENCE_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": encode_inputs},
        "5": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["4", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": STYLE_REFERENCE_LORA, "strength_model": strength},
        },
        "8": {
            "class_type": "ModelSamplingFlux",
            "inputs": {"model": ["7", 0], "max_shift": 1.15, "base_shift": 0.5, "width": width, "height": height},
        },
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "CFGGuider", "inputs": {"model": ["8", 0], "positive": ["5", 0], "negative": ["6", 0], "cfg": 1.0}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "BasicScheduler", "inputs": {"model": ["8", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["13", 0], "latent_image": ["9", 0]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["15", 0]}},
    }
    for index, image_name in enumerate(image_names, start=1):
        node_id = str(79 + index)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        encode_inputs[f"image{index}"] = [node_id, 0]
    return graph


def reid_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    reference_name: str,
    steps: int,
    styles: list[StyleSelection],
    user_loras: list[UserLoRASelection],
) -> dict[str, Any]:
    """Official Krea 2 ReID graph for independent character-scene generation."""
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": IDENTITY_EDIT_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": VISION_TEXT_ENCODER, "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": reference_name}},
        "5": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["4", 0],
                "upscale_method": "area",
                "megapixels": 0.140625,
                "resolution_steps": 16,
            },
        },
        "6": {
            "class_type": "TextEncodeKrea2OstrisEdit",
            "inputs": {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0], "image1": ["5", 0]},
        },
        "7": {
            "class_type": "TextEncodeKrea2OstrisEdit",
            "inputs": {"clip": ["2", 0], "prompt": "", "vae": ["3", 0], "image1": ["5", 0]},
        },
        "8": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "9": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["7", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "10": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": REID_LORA, "strength_model": 1.0},
        },
        "11": {
            "class_type": "Krea2OstrisEditModelPatch",
            "inputs": {"model": ["10", 0], "kv_cache": True},
        },
        "12": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "13": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["11", 0],
                "positive": ["8", 0],
                "negative": ["9", 0],
                "latent_image": ["12", 0],
                "seed": seed,
                "steps": max(8, steps),
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["14", 0]}},
    }
    model_input: list[Any] = ["10", 0]
    next_id = 20
    for style in styles:
        graph[str(next_id)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_input,
                "lora_name": STYLE_LORAS[style.name],
                "strength_model": style.strength,
            },
        }
        model_input = [str(next_id), 0]
        next_id += 1
    for selection in user_loras:
        graph[str(next_id)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_input,
                "lora_name": f"user/{selection.filename}",
                "strength_model": selection.strength,
            },
        }
        model_input = [str(next_id), 0]
        next_id += 1
    graph["11"]["inputs"]["model"] = model_input
    return graph


def character_sheet_workflow(reference_name: str, seed: int, prefix: str, steps: int) -> dict[str, Any]:
    """Experimental CharacterSheet QuadView candidate; never replaces ReID automatically."""
    prompt = (
        "Convert the character in the image to a character sheet showing a face close-up, "
        "front full body, side full body and back full body views"
    )
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": IDENTITY_EDIT_MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": reference_name}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": CHARACTER_SHEET_LORA, "strength_model": 1.0}},
        "7": {"class_type": "Krea2EditModelPatch", "inputs": {"model": ["6", 0], "source_latent": ["5", 0], "ref_boost": 1.0, "ref_boost_a": 1.0, "fit_mode": "fit", "vae": ["3", 0], "source_image": ["4", 0]}},
        "8": {"class_type": "Krea2EditGroundedEncode", "inputs": {"clip": ["2", 0], "prompt": prompt, "grounding_px": 0, "system_prompt": "", "image": ["4", 0]}},
        "9": {"class_type": "Krea2EditGroundedEncode", "inputs": {"clip": ["2", 0], "prompt": "", "grounding_px": 768, "system_prompt": "", "image": ["4", 0]}},
        "10": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1536, "height": 1024, "batch_size": 1}},
        "11": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["10", 0], "seed": seed, "steps": max(10, steps), "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["12", 0]}},
    }


def detail_enhance_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    source_name: str,
    strength: float,
    steps: int,
    vae_name: str,
    diffusion_model: str,
) -> dict[str, Any]:
    """Ostris edit graph used by the experimental Krea detail-enhancer LoRA."""
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_model, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": VISION_TEXT_ENCODER, "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "4": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        "5": {
            "class_type": "TextEncodeKrea2OstrisEdit",
            "inputs": {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0], "image1": ["4", 0]},
        },
        "6": {
            "class_type": "TextEncodeKrea2OstrisEdit",
            "inputs": {"clip": ["2", 0], "prompt": "", "vae": ["3", 0], "image1": ["4", 0]},
        },
        "7": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["5", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "8": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "9": {
            "class_type": "Krea2OstrisEditModelPatch",
            "inputs": {"model": ["1", 0], "kv_cache": False},
        },
        "10": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["9", 0], "lora_name": DETAIL_ENHANCER_LORA, "strength_model": strength},
        },
        "11": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["11", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "er_sde",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["13", 0]}},
    }
    return graph


def nk2e_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    reference_name: str,
    mode: str,
    strength: float,
    steps: int,
    diffusion_model: str,
) -> dict[str, Any]:
    graph = workflow(prompt, width, height, seed, prefix, steps, diffusion_model=diffusion_model)
    graph.update(
        {
            "10": {"class_type": "LoadImage", "inputs": {"image": reference_name}},
            "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["3", 0]}},
            "12": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["1", 0],
                    "lora_name": NK2E_CANNY_LORA if mode == "canny" else NK2E_EDIT_LORA,
                    "strength_model": strength,
                },
            },
            "13": {"class_type": "NK2EInContextModelNode", "inputs": {"model": ["12", 0]}},
            "14": {
                "class_type": "NK2ESetReferenceNode",
                "inputs": {"conditioning": ["4", 0], "reference": ["11", 0]},
            },
        }
    )
    graph["7"]["inputs"].update({"model": ["13", 0], "positive": ["14", 0]})
    return graph


def anypaint_workflow(
    prompt: str,
    seed: int,
    prefix: str,
    source_name: str,
    mask_name: str | None,
    left: int,
    top: int,
    right: int,
    bottom: int,
    strength: float,
    reference_max_edge: int,
    boundary_redraw_px: int,
    vlm_reference: bool,
    steps: int,
    diffusion_model: str,
) -> dict[str, Any]:
    prepare_inputs: dict[str, Any] = {
        "source": ["10", 0],
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "reference_max_edge": reference_max_edge,
        "boundary_redraw_px": boundary_redraw_px,
    }
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_model, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "10": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        "20": {"class_type": "Krea2AnyPaintPrepare", "inputs": prepare_inputs},
        "21": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": ANYPAINT_LORA, "strength_model": strength},
        },
        "22": {"class_type": "Krea2AnyPaintModelPatch", "inputs": {"model": ["21", 0], "kv_cache": True}},
        "23": {
            "class_type": "Krea2AnyPaintEncode",
            "inputs": {
                "clip": ["2", 0],
                "prompt": prompt,
                "vae": ["3", 0],
                "semantic_reference": ["20", 0],
                "known_image": ["20", 1],
                "keep_mask": ["20", 3],
                "vlm_reference": vlm_reference,
            },
        },
        "24": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "25": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["22", 0],
                "positive": ["23", 0],
                "negative": ["24", 0],
                "latent_image": ["23", 1],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["3", 0]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["26", 0]}},
    }
    if mask_name is not None:
        graph["11"] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}
        graph["12"] = {"class_type": "ImageToMask", "inputs": {"image": ["11", 0], "channel": "red"}}
        prepare_inputs["generated_mask"] = ["12", 0]
    return graph


def depth_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    depth_name: str,
    control_strength: float,
    steps: int,
    styles: list[StyleSelection],
    user_loras: list[UserLoRASelection],
    diffusion_model: str,
) -> dict[str, Any]:
    graph = workflow(prompt, width, height, seed, prefix, steps, styles, user_loras, diffusion_model)
    model_input = graph["7"]["inputs"]["model"]
    graph.update(
        {
            "10": {"class_type": "LoadImage", "inputs": {"image": depth_name}},
            "11": {
                "class_type": "Krea2ControlLoRALoader",
                "inputs": {
                    "model": model_input,
                    "lora_name": DEPTH_CONTROL_LORA,
                    "strength": control_strength,
                },
            },
            "12": {
                "class_type": "Krea2ControlImageEncode",
                "inputs": {
                    "control_image": ["10", 0],
                    "vae": ["3", 0],
                    "resize": "match_latent_size",
                    "upscale_method": "lanczos",
                    "crop": "center",
                    "channel_mode": "grayscale",
                    "normalize": "per_image_minmax",
                    "invert": False,
                    "batch_mode": "independent_images",
                    "latent": ["6", 0],
                },
            },
            "13": {
                "class_type": "Krea2ControlApply",
                "inputs": {"model": ["11", 0], "control_latent": ["12", 0]},
            },
        }
    )
    graph["7"]["inputs"]["model"] = ["13", 0]
    return graph


def identity_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    source_name: str,
    reference_names: list[str],
    identity_mask_name: str | None,
    identity_strength: float,
    ref_boost: float,
    source_ref_boost: float,
    grounding_px: int,
    steps: int,
    styles: list[StyleSelection],
    user_loras: list[UserLoRASelection],
    depth_name: str | None,
    control_strength: float,
    fit_mode: str,
    diffusion_model: str,
    text_encoder: str = TEXT_ENCODER,
    apply_identity_lora: bool = True,
    identity_preset: str = "",
) -> dict[str, Any]:
    graph = workflow(prompt, width, height, seed, prefix, steps, diffusion_model=diffusion_model)
    graph["2"]["inputs"]["clip_name"] = text_encoder
    functional_lora = HEAD_SWAP_LORA if identity_preset == "headSwap" else IDENTITY_EDIT_LORA
    graph.update(
        {
            "10": {"class_type": "LoadImage", "inputs": {"image": source_name}},
            "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["3", 0]}},
            "12": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "13": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["1", 0],
                    "lora_name": functional_lora,
                    "strength_model": identity_strength,
                },
            },
            "15": {
                "class_type": "Krea2EditModelPatch",
                "inputs": {
                    "model": ["13", 0],
                    "source_latent": ["11", 0],
                    "ref_boost": ref_boost,
                    "ref_boost_a": source_ref_boost,
                    "fit_mode": "crop (legacy)" if fit_mode == "crop" else "fit",
                    "vae": ["3", 0],
                    "source_image": ["10", 0],
                },
            },
            "16": {
                "class_type": "Krea2EditGroundedEncode",
                "inputs": {
                    "clip": ["2", 0],
                    "prompt": prompt,
                    "grounding_px": grounding_px,
                    "system_prompt": "",
                    "image": ["10", 0],
                },
            },
            "17": {
                "class_type": "Krea2EditGroundedEncode",
                "inputs": {
                    "clip": ["2", 0],
                    "prompt": "",
                    "grounding_px": grounding_px,
                    "system_prompt": "",
                    "image": ["10", 0],
                },
            },
        }
    )
    # The dedicated BFS head-swap graph uses its LoRA by itself.  The
    # checkpoint-specific Identity Edit exception must not remove it.
    if apply_identity_lora or identity_preset == "headSwap":
        model_input: list[Any] = ["13", 0]
    else:
        graph.pop("13")
        graph["15"]["inputs"]["model"] = ["1", 0]
        model_input = ["1", 0]
    next_id = 20
    for style in styles:
        graph[str(next_id)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_input,
                "lora_name": STYLE_LORAS[style.name],
                "strength_model": style.strength,
            },
        }
        model_input = [str(next_id), 0]
        next_id += 1
    for selection in user_loras:
        graph[str(next_id)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_input,
                "lora_name": f"user/{selection.filename}",
                "strength_model": selection.strength,
            },
        }
        model_input = [str(next_id), 0]
        next_id += 1
    if depth_name is not None:
        graph[str(next_id)] = {"class_type": "LoadImage", "inputs": {"image": depth_name}}
        depth_load_id = str(next_id)
        next_id += 1
        graph[str(next_id)] = {
            "class_type": "Krea2ControlLoRALoader",
            "inputs": {
                "model": model_input,
                "lora_name": DEPTH_CONTROL_LORA,
                "strength": control_strength,
            },
        }
        depth_lora_id = str(next_id)
        next_id += 1
        graph[str(next_id)] = {
            "class_type": "Krea2ControlImageEncode",
            "inputs": {
                "control_image": [depth_load_id, 0],
                "vae": ["3", 0],
                "resize": "match_latent_size",
                "upscale_method": "lanczos",
                "crop": "center",
                "channel_mode": "grayscale",
                "normalize": "per_image_minmax",
                "invert": False,
                "batch_mode": "independent_images",
                "latent": ["12", 0],
            },
        }
        depth_encode_id = str(next_id)
        next_id += 1
        graph[str(next_id)] = {
            "class_type": "Krea2ControlApply",
            "inputs": {"model": [depth_lora_id, 0], "control_latent": [depth_encode_id, 0]},
        }
        model_input = [str(next_id), 0]
    graph["15"]["inputs"]["model"] = model_input

    if reference_names:
        # Keep the published graph topology intact.  In particular, outfit and
        # pose references are loaded separately and joined by ComfyUI's native
        # ImageStitch node.  Pre-compositing them in Pillow was visually close,
        # but was not byte/geometry-equivalent to the workflow Identity Edit was
        # demonstrated with.
        load_ids: list[str] = []
        for index, reference_name in enumerate(reference_names):
            load_id = str(180 + index)
            graph[load_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": reference_name},
            }
            load_ids.append(load_id)
        reference_output: list[Any] = [load_ids[0], 0]
        for index, load_id in enumerate(load_ids[1:]):
            stitch_id = str(200 + index)
            graph[stitch_id] = {
                "class_type": "ImageStitch",
                "inputs": {
                    "image1": reference_output,
                    "image2": [load_id, 0],
                    "direction": "right",
                    "match_image_size": True,
                    "spacing_width": 0,
                    "spacing_color": "white",
                },
            }
            reference_output = [stitch_id, 0]
        graph["280"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": reference_output, "vae": ["3", 0]},
        }
        graph["15"]["inputs"].update(
            {"source_latent_b": ["280", 0], "source_image_b": reference_output}
        )
        graph["16"]["inputs"]["image_b"] = reference_output
        graph["17"]["inputs"]["image_b"] = reference_output

    if identity_mask_name is not None:
        graph["14"] = {
            "class_type": "LoadImageMask",
            "inputs": {"image": identity_mask_name, "channel": "red"},
        }
        graph["15"]["inputs"]["ref_boost_mask"] = ["14", 0]

    graph["7"]["inputs"].update(
        {
            "model": ["15", 0],
            "positive": ["16", 0],
            "negative": ["17", 0],
            "latent_image": ["12", 0],
        }
    )
    return graph


def decode_image(encoded: str, preserve_alpha: bool = False) -> Image.Image:
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image must be valid base64") from exc
    if len(raw) > 32 << 20:
        raise HTTPException(status_code=400, detail="image exceeds 32 MiB")
    try:
        image = Image.open(io.BytesIO(raw))
        if preserve_alpha and ("A" in image.getbands() or "transparency" in image.info):
            return image.convert("RGBA")
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="image is not a valid image") from exc


def make_depth_image(source: Image.Image) -> Image.Image:
    global depth_processor, depth_model

    import torch
    import torch.nn.functional as functional
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if depth_processor is None or depth_model is None:
        depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).eval()

    inputs = depth_processor(images=source, return_tensors="pt")
    with torch.inference_mode():
        prediction = depth_model(**inputs).predicted_depth
    prediction = functional.interpolate(
        prediction.unsqueeze(1),
        size=(source.height, source.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    prediction -= prediction.min()
    prediction /= prediction.max().clamp_min(1e-6)
    gray = (prediction.mul(255).byte().cpu().numpy())
    return Image.fromarray(gray, mode="L").convert("RGB")


def save_depth_input(encoded: str) -> tuple[str, Path, str]:
    source = decode_image(encoded)
    depth = make_depth_image(source)
    relative = f"krea-depth/{uuid.uuid4().hex}.png"
    path = INPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    depth.save(path, format="PNG")
    preview = io.BytesIO()
    depth.save(preview, format="PNG")
    return relative, path, base64.b64encode(preview.getvalue()).decode("ascii")


def save_nk2e_input(encoded: str, mode: str, preprocessed: bool = False) -> tuple[str, Path, str | None]:
    image = decode_image(encoded)
    preview_encoded: str | None = None
    if mode == "canny" and not preprocessed:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        image = Image.fromarray(edges, mode="L").convert("RGB")
        preview = io.BytesIO()
        image.save(preview, format="PNG")
        preview_encoded = base64.b64encode(preview.getvalue()).decode("ascii")
    relative = f"nk2e-{mode}/{uuid.uuid4().hex}.png"
    path = INPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    if mode == "canny" and preprocessed:
        preview = io.BytesIO()
        image.save(preview, format="PNG")
        preview_encoded = base64.b64encode(preview.getvalue()).decode("ascii")
    return relative, path, preview_encoded


def composite_strict_mask(encoded: str, source_path: Path, mask_path: Path, grow: int, feather: float) -> str:
    generated = decode_image(encoded)
    source = Image.open(source_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if generated.size != source.size or mask.size != source.size:
        raise HTTPException(status_code=400, detail="strict mask, source, and generated image dimensions must match")
    if grow:
        mask = mask.filter(ImageFilter.MaxFilter(grow * 2 + 1))
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    result = Image.composite(generated, source, mask)
    output = io.BytesIO()
    result.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def save_input(encoded: str, folder: str) -> tuple[str, Path]:
    image = decode_image(encoded)
    relative = f"{folder}/{uuid.uuid4().hex}.png"
    path = INPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return relative, path


def prepare_identity_reference(encoded: str) -> Image.Image:
    """Turn an RGBA cutout into the opaque product-style image source B expects.

    ComfyUI's LoadImage returns RGB pixels and a separate mask. Identity Edit only
    consumes the RGB output, so simply passing an extracted RGBA garment exposes
    the original person's RGB pixels hidden below alpha=0. Crop the visible object,
    enlarge it onto a neutral canvas, and bake alpha before VAE encoding.
    """
    image = decode_image(encoded, preserve_alpha=True)
    if image.mode != "RGBA":
        return image.convert("RGB")

    alpha = image.getchannel("A")
    minimum, _ = alpha.getextrema()
    if minimum == 255:
        return image.convert("RGB")
    visible = alpha.point(lambda value: 255 if value > 4 else 0)
    bounds = visible.getbbox()
    if bounds is None:
        raise HTTPException(status_code=400, detail="identity reference has no visible pixels")

    left, top, right, bottom = bounds
    padding = max(8, round(max(right - left, bottom - top) * 0.06))
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    cropped = image.crop((left, top, right, bottom))

    neutral = (245, 245, 245, 255)
    flattened = Image.new("RGBA", cropped.size, neutral)
    flattened.alpha_composite(cropped)
    flattened_rgb = flattened.convert("RGB")

    fit_width = max(1, round(image.width * 0.88))
    fit_height = max(1, round(image.height * 0.88))
    fitted = ImageOps.contain(flattened_rgb, (fit_width, fit_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", image.size, neutral[:3])
    canvas.paste(fitted, ((image.width - fitted.width) // 2, (image.height - fitted.height) // 2))
    return canvas


def save_identity_reference(encoded: str) -> tuple[str, Path]:
    image = prepare_identity_reference(encoded)
    relative = f"krea-edit-reference/{uuid.uuid4().hex}.png"
    path = INPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return relative, path


def save_stitched_references(encoded_images: list[str]) -> tuple[str, Path]:
    """Build the single source-B image expected by Krea2 Edit from several references."""
    images = [decode_image(encoded).convert("RGB") for encoded in encoded_images]
    if not images:
        raise ValueError("at least one reference image is required")
    if len(images) == 1:
        return save_input(encoded_images[0], "krea-edit-reference")

    # Match ComfyUI's ImageStitch(direction=right, match_image_size=true)
    # exactly: keep image one unchanged, resize each following image to image
    # one's height with its aspect ratio intact, then concatenate without gaps.
    # The previous equal-panel/padding implementation changed the source-B
    # geometry from the published workflow and weakened the edit references.
    base_height = images[0].height
    fitted: list[Image.Image] = [images[0]]
    for image in images[1:]:
        target_width = max(1, int(base_height * image.width / image.height))
        fitted.append(image.resize((target_width, base_height), Image.Resampling.LANCZOS))
    stitched = Image.new("RGB", (sum(image.width for image in fitted), base_height), "white")
    offset = 0
    for image in fitted:
        stitched.paste(image, (offset, 0))
        offset += image.width
    relative = f"krea-edit-reference/{uuid.uuid4().hex}.png"
    path = INPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    stitched.save(path, format="PNG")
    return relative, path


async def comfy_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"{COMFY_URL}/object_info")
        return response.is_success
    except httpx.HTTPError:
        return False


@app.post("/v1/cancel")
async def cancel_generation() -> dict[str, str]:
    """Interrupt the single active ComfyUI generation, if any."""
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.post(f"{COMFY_URL}/interrupt")
        response.raise_for_status()
    return {"status": "cancelling"}


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
            for output in history.get("outputs", {}).values():
                images = output.get("images", [])
                if images:
                    return images[0]
        await asyncio.sleep(0.25)
    raise TimeoutError("image generation timed out")


async def execute_workflow(graph: dict[str, Any], operation_id: str = "") -> str:
    image: dict[str, Any] | None = None
    try:
        client_id = uuid.uuid4().hex
        ws_url = COMFY_URL.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={client_id}"
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, heartbeat=30, receive_timeout=30 * 60) as websocket:
                async with httpx.AsyncClient(timeout=30 * 60) as client:
                    submitted = await client.post(
                        f"{COMFY_URL}/prompt", json={"prompt": graph, "client_id": client_id}
                    )
                    submitted.raise_for_status()
                    body = submitted.json()
                    if body.get("node_errors"):
                        raise RuntimeError(f"invalid workflow: {body['node_errors']}")
                    prompt_id = body["prompt_id"]
                    while True:
                        message = await websocket.receive()
                        if message.type == aiohttp.WSMsgType.ERROR:
                            raise RuntimeError(f"ComfyUI websocket failed: {websocket.exception()}")
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        event = json.loads(message.data)
                        data = event.get("data") or {}
                        if data.get("prompt_id") != prompt_id:
                            continue
                        event_type = event.get("type")
                        if event_type == "execution_error":
                            raise RuntimeError(f"ComfyUI generation failed: {data}")
                        if event_type == "progress":
                            current = float(data.get("value") or 0)
                            total = max(1.0, float(data.get("max") or 1))
                            ratio = max(0.0, min(1.0, current / total))
                            publish_runtime_operation(
                                operation_id,
                                "sampling",
                                "Krea sampler",
                                f"이미지 확산 추론 {int(current)}/{int(total)}",
                                0.55 + ratio * 0.30,
                                "",
                            )
                            continue
                        if event_type == "executing":
                            node_id = data.get("node")
                            if node_id is None:
                                break
                            phase, component, detail, progress, memory_action = comfy_node_runtime_phase(
                                graph, str(node_id)
                            )
                            publish_runtime_operation(
                                operation_id, phase, component, detail, progress, memory_action
                            )
                    image = await wait_for_output(client, prompt_id)
                    viewed = await client.get(f"{COMFY_URL}/view", params=image)
                    viewed.raise_for_status()
                    return base64.b64encode(viewed.content).decode("ascii")
    finally:
        if image is not None:
            candidate = (OUTPUT_ROOT / image["subfolder"] / image["filename"]).resolve()
            if candidate.is_relative_to(OUTPUT_ROOT):
                candidate.unlink(missing_ok=True)


def checkpoint_status() -> dict[str, Any]:
    variants = []
    for checkpoint in CHECKPOINTS.values():
        is_ready = ready(checkpoint)
        if is_ready:
            try:
                link_model(checkpoint)
            except RuntimeError:
                pass
        partial = model_path(checkpoint).with_suffix(".safetensors.part")
        downloaded = checkpoint.size if is_ready else partial.stat().st_size if partial.is_file() else 0
        variants.append(
            {
                "id": checkpoint.key,
                "label": checkpoint.label,
                "precision": checkpoint.precision,
                "ready": is_ready,
                "downloaded_bytes": downloaded,
                "size_bytes": checkpoint.size,
                "source": checkpoint.source,
                "recommended_sampler": CHECKPOINT_SAMPLING.get(checkpoint.key, ("euler", "simple"))[0],
                "recommended_scheduler": CHECKPOINT_SAMPLING.get(checkpoint.key, ("euler", "simple"))[1],
            }
        )
    heretic_ready = ready(HERETIC_TEXT_ENCODER)
    if heretic_ready:
        try:
            link_text_encoder()
        except RuntimeError:
            pass
    heretic_partial = model_path(HERETIC_TEXT_ENCODER).with_suffix(".safetensors.part")
    return {
        "ready": all(item["ready"] for item in variants),
        "preparing": checkpoint_prepare_task is not None and not checkpoint_prepare_task.done(),
        "token_configured": token_configured(),
        "current": checkpoint_prepare_current,
        "downloaded_bytes": checkpoint_prepare_bytes,
        "total_bytes": checkpoint_prepare_total,
        "error": checkpoint_prepare_error,
        "variants": variants,
        "identity_runtime": {
            "convrot_ready": (Path("/opt/ComfyUI/models/diffusion_models") / IDENTITY_EDIT_MODEL).is_file(),
            "convrot_source": "https://huggingface.co/Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8",
            "heretic_ready": heretic_ready,
            "heretic_downloaded_bytes": HERETIC_TEXT_ENCODER.size if heretic_ready else heretic_partial.stat().st_size if heretic_partial.is_file() else 0,
            "heretic_size_bytes": HERETIC_TEXT_ENCODER.size,
            "heretic_source": HERETIC_TEXT_ENCODER.source,
        },
        "nvfp4_conversion": {
            "available": True,
            "preparing": checkpoint_conversion_task is not None and not checkpoint_conversion_task.done(),
            "current": checkpoint_conversion_current,
            "stage": checkpoint_conversion_stage,
            "done": checkpoint_conversion_done,
            "total": checkpoint_conversion_total,
            "error": checkpoint_conversion_error,
            "profile_source": PROFILE_SOURCE,
            "profile_commit": PROFILE_COMMIT,
            "variants": [
                {
                    "id": key,
                    "source_ready": ready(source),
                    "source_size_bytes": source.size,
                    "converted_ready": nvfp4_ready(key),
                    "validated": nvfp4_validated(key),
                    "converted_size_bytes": nvfp4_path(key).stat().st_size if nvfp4_ready(key) else 0,
                    "source": source.source,
                }
                for key, source in BF16_SOURCES.items()
            ],
        },
    }


async def _prepare_checkpoints(keys: list[str], civitai_token: str, hf_token: str) -> None:
    global checkpoint_prepare_bytes, checkpoint_prepare_current, checkpoint_prepare_error, checkpoint_prepare_total
    async with checkpoint_prepare_lock:
        checkpoint_prepare_error = ""
        checkpoint_prepare_bytes = 0
        items = [HERETIC_TEXT_ENCODER, *(CHECKPOINTS[key] for key in keys)]
        checkpoint_prepare_total = sum(item.size for item in items)
        completed = 0
        try:
            for checkpoint in items:
                checkpoint_prepare_current = checkpoint.key

                def update(done: int, _total: int, base: int = completed) -> None:
                    global checkpoint_prepare_bytes
                    checkpoint_prepare_bytes = base + done

                download_token = hf_token if checkpoint.provider == "huggingface" else civitai_token
                await asyncio.to_thread(download_checkpoint, checkpoint, download_token, update, checkpoint is not HERETIC_TEXT_ENCODER)
                if checkpoint is HERETIC_TEXT_ENCODER:
                    await asyncio.to_thread(link_text_encoder)
                completed += checkpoint.size
                checkpoint_prepare_bytes = completed
        except BaseException as exc:
            checkpoint_prepare_error = str(exc)
        finally:
            checkpoint_prepare_current = ""


@app.get("/v1/checkpoints/status")
async def get_checkpoint_status() -> dict[str, Any]:
    return checkpoint_status()


@app.post("/v1/checkpoints/prepare", status_code=202)
async def prepare_checkpoints(request: CheckpointPrepareRequest) -> dict[str, Any]:
    global checkpoint_prepare_task
    token = request.civitai_token.strip()
    if token:
        if len(token) < 16 or any(character.isspace() for character in token):
            raise HTTPException(status_code=400, detail="invalid Civitai API key")
        save_token(token)
    effective_token = token or stored_token()
    if not effective_token:
        raise HTTPException(status_code=400, detail="enter a Civitai API key")
    hf_token = request.hf_token.strip()
    if hf_token:
        _save_hf_token(hf_token)
    effective_hf_token = hf_token or _stored_hf_token()
    keys = list(dict.fromkeys(request.variants))
    unknown = [key for key in keys if key not in CHECKPOINTS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unsupported checkpoint variants: {unknown}")
    if not keys:
        raise HTTPException(status_code=400, detail="select at least one checkpoint")
    if checkpoint_prepare_task is None or checkpoint_prepare_task.done():
        checkpoint_prepare_task = asyncio.create_task(_prepare_checkpoints(keys, effective_token, effective_hf_token))
        started = True
    else:
        started = False
    return {**checkpoint_status(), "started": started}


async def _unload_comfy_models() -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{COMFY_URL}/free",
            json={"unload_models": True, "free_memory": True},
        )
        response.raise_for_status()


async def _convert_checkpoints(keys: list[str], token: str, remove_sources: bool) -> None:
    global checkpoint_conversion_current, checkpoint_conversion_done, checkpoint_conversion_error
    global checkpoint_conversion_stage, checkpoint_conversion_total
    checkpoint_conversion_error = ""
    checkpoint_conversion_done = 0
    checkpoint_conversion_total = sum(BF16_SOURCES[key].size for key in keys)
    try:
        for key in keys:
            source = BF16_SOURCES[key]
            checkpoint_conversion_current = key
            checkpoint_conversion_stage = "download"
            checkpoint_conversion_total = source.size
            checkpoint_conversion_done = 0

            def download_progress(done: int, _total: int) -> None:
                global checkpoint_conversion_done
                checkpoint_conversion_done = done

            await asyncio.to_thread(download_checkpoint, source, token, download_progress, False)
            checkpoint_conversion_done = source.size
            async with generation_lock:
                result: dict[str, int]
                if nvfp4_ready(key):
                    try:
                        result = json.loads(nvfp4_validation_path(key).read_text())
                    except (OSError, ValueError):
                        result = {"source_tensors": 0, "output_tensors": 0, "quantized_layers": 0}
                else:
                    checkpoint_conversion_stage = "unload"
                    await _unload_comfy_models()
                    checkpoint_conversion_stage = "convert"
                    checkpoint_conversion_done = 0
                    checkpoint_conversion_total = 1

                    def quant_progress(done: int, total: int) -> None:
                        global checkpoint_conversion_done, checkpoint_conversion_total
                        checkpoint_conversion_done = done
                        checkpoint_conversion_total = total

                    result = await asyncio.to_thread(
                        convert_krea2_nvfp4,
                        model_path(source),
                        nvfp4_path(key),
                        quant_progress,
                    )
                link_nvfp4(key)
                checkpoint_conversion_stage = "validate"
                validation_graph = workflow(
                    "a blue ceramic cup on a plain wooden table",
                    512,
                    512,
                    24681357,
                    f"nvfp4-validation/{key}",
                    8,
                    diffusion_model=NVFP4_FILENAMES[key],
                )
                await execute_workflow(validation_graph)
                nvfp4_validation_path(key).write_text(json.dumps(result, sort_keys=True))
                if remove_sources:
                    model_path(source).unlink(missing_ok=True)
                await _unload_comfy_models()
                checkpoint_conversion_done = checkpoint_conversion_total
    except BaseException as exc:
        checkpoint_conversion_error = str(exc)
    finally:
        checkpoint_conversion_current = ""
        checkpoint_conversion_stage = ""


@app.post("/v1/checkpoints/convert-nvfp4", status_code=202)
async def convert_checkpoints(request: CheckpointConvertRequest) -> dict[str, Any]:
    global checkpoint_conversion_task
    token = request.civitai_token.strip()
    if token:
        if len(token) < 16 or any(character.isspace() for character in token):
            raise HTTPException(status_code=400, detail="invalid Civitai API key")
        save_token(token)
    effective_token = token or stored_token()
    if not effective_token:
        raise HTTPException(status_code=400, detail="enter a Civitai API key")
    keys = list(dict.fromkeys(request.variants))
    unknown = [key for key in keys if key not in BF16_SOURCES]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unsupported NVFP4 variants: {unknown}")
    if not keys:
        raise HTTPException(status_code=400, detail="select at least one NVFP4 variant")
    if checkpoint_prepare_task is not None and not checkpoint_prepare_task.done():
        raise HTTPException(status_code=409, detail="checkpoint download is already running")
    if checkpoint_conversion_task is None or checkpoint_conversion_task.done():
        checkpoint_conversion_task = asyncio.create_task(
            _convert_checkpoints(keys, effective_token, request.remove_bf16_sources)
        )
        started = True
    else:
        started = False
    return {**checkpoint_status(), "started": started}


@app.get("/health")
async def health() -> dict[str, Any]:
    if not await comfy_ready():
        raise HTTPException(status_code=503, detail="NVFP4 runtime is starting")
    return {
        "status": "ok",
        "busy": generation_lock.locked(),
        "segmenting": segmentation_lock.locked(),
        "runtime": runtime_status(),
    }


def request_runtime_profile(request: ImageRequest) -> str:
    if request.runtime_profile:
        return request.runtime_profile
    if request.detail_enhance_image:
        return "krea-detail"
    if request.anypaint_image:
        return "krea-anypaint"
    if request.nk2e_image:
        return f"krea-nk2e-{request.nk2e_mode}"
    if request.style_reference_images:
        return "krea-style-reference"
    if request.character_sheet_image:
        return "krea-character-sheet"
    if request.reid_image:
        return "krea-reid"
    if request.source_image:
        if request.identity_preset == "headSwap":
            return "krea-head-swap"
        return f"krea-identity-{request.identity_model}-{request.identity_encoder}"
    if request.control_image:
        return "krea-depth"
    if request.vision_images:
        return f"krea-vision-{request.vision_mode}"
    return "krea-create"


def request_runtime_signature(request: ImageRequest) -> str:
    value = {
        "profile": request_runtime_profile(request),
        "checkpoint": request.checkpoint,
        "reid": bool(request.reid_image),
        "character_sheet": bool(request.character_sheet_image),
        "identity_model": request.identity_model if request.source_image else "",
        "identity_encoder": request.identity_encoder if request.source_image else "",
        "identity_preset": request.identity_preset if request.source_image else "",
        "vae": request.detail_vae if request.detail_enhance_image else request.vae_mode,
        "styles": [(item.name, round(item.strength, 4)) for item in request.styles],
        "user_loras": [(item.filename, round(item.strength, 4)) for item in request.user_loras],
        "filter": (request.filter_mode, request.filter_strength),
        "prompt_enhancer": (request.prompt_enhancer, round(request.prompt_enhancer_strength, 4)),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def publish_runtime_operation(
    operation_id: str,
    phase: str,
    component: str,
    detail: str,
    progress: float = 0.0,
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


def comfy_node_runtime_phase(graph: dict[str, Any], node_id: str) -> tuple[str, str, str, float, str]:
    class_type = str((graph.get(str(node_id)) or {}).get("class_type", ""))
    lowered = class_type.lower()
    if "loadimage" in lowered:
        return "preparing", class_type or "입력 이미지", "참조 이미지 로딩", 0.14, ""
    if any(token in lowered for token in ("loader", "loadmodel", "loadvae", "cliploader")):
        return "model_loading", class_type or "ComfyUI 모델", "모델·LoRA 가중치 탑재", 0.18, "load"
    if "vaeencode" in lowered:
        return "conditioning", class_type or "VAE 인코더", "참조 이미지를 잠재 조건으로 변환", 0.40, ""
    if any(token in lowered for token in ("textencode", "encode", "conditioning", "visualstyleref")) and "vae" not in lowered:
        return "conditioning", class_type or "조건 인코더", "프롬프트·참조 조건 인코딩", 0.32, ""
    if "modelpatch" in lowered:
        return "conditioning", class_type or "편집 모델", "참조 조건을 생성 모델에 적용", 0.48, ""
    if any(token in lowered for token in ("sampler", "sampling", "guider")):
        return "sampling", class_type or "Krea 샘플러", "이미지 확산 추론", 0.55, ""
    if "decode" in lowered or "vae" in lowered:
        return "decoding", class_type or "VAE", "잠재 이미지 디코딩", 0.88, ""
    if any(token in lowered for token in ("saveimage", "previewimage")):
        return "finalizing", class_type or "이미지 출력", "결과 이미지 저장", 0.96, ""
    return "sampling", class_type or "ComfyUI 그래프", "이미지 그래프 실행", 0.48, ""


def runtime_status() -> dict[str, Any]:
    elapsed = max(0.0, time.monotonic() - runtime_started_at) if runtime_started_at else 0.0
    return {
        "status": runtime_stage,
        "profile": runtime_profile,
        "signature": runtime_signature,
        "preparing": runtime_stage == "preparing",
        "elapsed_seconds": round(elapsed, 3) if runtime_stage in {"preparing", "generating"} else 0,
        "last_load_seconds": round(runtime_last_load_seconds, 3),
        "error": runtime_error,
        "operation": runtime_operation,
        "operation_history": runtime_operation_history,
    }


@app.get("/v1/models/runtime/status")
async def model_runtime_status(operation_id: str = "") -> dict[str, Any]:
    if not await comfy_ready():
        raise HTTPException(status_code=503, detail="NVFP4 runtime is starting")
    status = runtime_status()
    if operation_id and (not runtime_operation or runtime_operation.get("operation_id") != operation_id):
        status["operation"] = None
        status["operation_history"] = []
    return status


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


@app.get("/v1/loras")
async def loras() -> dict[str, Any]:
    """Return the allow-listed user LoRAs visible to generation requests."""
    data = []
    if USER_LORA_ROOT.is_dir():
        for path in sorted(USER_LORA_ROOT.glob("*.safetensors"), key=lambda item: item.name.lower()):
            # skc3vo is exposed through filter_mode=adherence, not as a
            # stackable user LoRA.
            if path.is_file() and path.name != "skc3vo.safetensors":
                data.append({"filename": path.name, "size": path.stat().st_size})
    return {"object": "list", "data": data}


@app.get("/v1/user-loras/status")
async def user_lora_status() -> dict[str, bool]:
    return {
        "civitai_token_configured": bool(stored_token()),
        "hf_token_configured": bool(_stored_hf_token()),
    }


@app.post("/v1/user-loras/tokens")
async def save_download_credentials(request: DownloadCredentialRequest) -> dict[str, bool]:
    if request.civitai_token.strip():
        save_token(request.civitai_token.strip())
    if request.hf_token.strip():
        _save_hf_token(request.hf_token.strip())
    return {
        "civitai_token_configured": bool(stored_token()),
        "hf_token_configured": bool(_stored_hf_token()),
    }


@app.get("/v1/user-loras")
async def list_user_loras() -> list[dict[str, Any]]:
    return _list_user_loras()


@app.post("/v1/user-loras/import", status_code=201)
async def import_user_lora(request: UserLoRAImportRequest) -> dict[str, Any]:
    source = request.source.strip()
    provider = request.provider.strip().lower()
    if provider == "auto":
        provider = "civitai" if source.isdigit() or "civitai." in source.lower() else "huggingface"
    if provider not in {"civitai", "huggingface"}:
        raise HTTPException(status_code=400, detail="지원하지 않는 LoRA 공급자입니다")
    try:
        metadata = await asyncio.to_thread(
            _import_civitai_lora if provider == "civitai" else _import_hf_lora, request
        )
        destination = USER_LORA_ROOT / metadata["filename"]
        _write_json(destination.with_suffix(".json"), metadata)
        return {**metadata, "size": destination.stat().st_size}
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/user-loras/upload", status_code=201)
async def upload_user_lora(
    file: UploadFile = File(...),
    name: str = Form(default=""),
    trigger_word: str = Form(default=""),
    memo: str = Form(default=""),
    base_model: str = Form(default=""),
    recommended_strength: float = Form(default=1.0),
) -> dict[str, Any]:
    original_name = Path(file.filename or "").name
    if not original_name.lower().endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="safetensors LoRA 파일만 업로드할 수 있습니다")
    if not -2.0 <= recommended_strength <= 2.0:
        raise HTTPException(status_code=400, detail="기본 강도는 -2.00부터 2.00까지입니다")
    filename = _safe_lora_filename(name.strip() or original_name)
    destination = USER_LORA_ROOT / filename
    if destination.exists():
        raise HTTPException(status_code=409, detail=f"{filename}이 이미 등록되어 있습니다")
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_USER_LORA_BYTES:
                    raise HTTPException(status_code=413, detail="LoRA 파일이 2 GiB 제한을 초과합니다")
                output.write(chunk)
                digest.update(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="빈 파일은 등록할 수 없습니다")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
        await file.close()
    metadata = {
        "filename": filename,
        "name": name.strip() or Path(filename).stem,
        "trigger_word": trigger_word.strip(),
        "memo": memo.strip()[:2000],
        "base_model": base_model.strip()[:128],
        "recommended_strength": recommended_strength,
        "source": "",
        "provider": "upload",
        "original_filename": original_name,
        "sha256": digest.hexdigest(),
        "created_at": time.time(),
    }
    _write_json(destination.with_suffix(".json"), metadata)
    return {**metadata, "size": total}


@app.put("/v1/user-loras/{filename}")
async def update_user_lora(filename: str, request: UserLoRAUpdateRequest) -> dict[str, Any]:
    path = _user_lora_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="LoRA를 찾지 못했습니다")
    metadata = _read_json(path.with_suffix(".json"))
    metadata.update({
        "name": request.name.strip() or path.stem,
        "trigger_word": request.trigger_word.strip(),
        "memo": request.memo.strip(),
        "base_model": request.base_model.strip(),
        "recommended_strength": request.recommended_strength,
        "updated_at": time.time(),
    })
    _write_json(path.with_suffix(".json"), metadata)
    return {**metadata, "filename": path.name, "size": path.stat().st_size}


@app.get("/v1/user-loras/{filename}/preview")
async def user_lora_preview(filename: str) -> FileResponse:
    path = _user_lora_preview_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="대표 이미지를 찾지 못했습니다")
    return FileResponse(path, media_type="image/webp", headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.put("/v1/user-loras/{filename}/preview")
async def update_user_lora_preview(filename: str, file: UploadFile = File(...)) -> dict[str, Any]:
    lora_path = _user_lora_path(filename)
    if not lora_path.is_file():
        raise HTTPException(status_code=404, detail="LoRA를 찾지 못했습니다")
    raw = await file.read(MAX_USER_LORA_PREVIEW_BYTES + 1)
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="빈 이미지는 등록할 수 없습니다")
    if len(raw) > MAX_USER_LORA_PREVIEW_BYTES:
        raise HTTPException(status_code=413, detail="대표 이미지는 20 MiB 이하여야 합니다")
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="지원하는 이미지 파일이 아닙니다") from exc
    destination = _user_lora_preview_path(filename)
    temporary = destination.with_suffix(".webp.part")
    try:
        image.save(temporary, format="WEBP", quality=88, method=6)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    metadata_path = lora_path.with_suffix(".json")
    metadata = _read_json(metadata_path)
    metadata["preview_updated_at"] = time.time()
    _write_json(metadata_path, metadata)
    return {"preview_available": True, "preview_updated_at": metadata["preview_updated_at"]}


@app.delete("/v1/user-loras/{filename}/preview", status_code=204)
async def delete_user_lora_preview(filename: str) -> None:
    lora_path = _user_lora_path(filename)
    if not lora_path.is_file():
        raise HTTPException(status_code=404, detail="LoRA를 찾지 못했습니다")
    _user_lora_preview_path(filename).unlink(missing_ok=True)
    metadata_path = lora_path.with_suffix(".json")
    metadata = _read_json(metadata_path)
    metadata.pop("preview_updated_at", None)
    _write_json(metadata_path, metadata)


@app.delete("/v1/user-loras/{filename}", status_code=204)
async def delete_user_lora(filename: str) -> None:
    path = _user_lora_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="LoRA를 찾지 못했습니다")
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    path.with_suffix(".preview.webp").unlink(missing_ok=True)


def _automatic_mask(request: SegmentRequest) -> dict[str, Any]:
    """Text-ground an object and refine its union mask with SAM 2.1."""
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    image = decode_image(request.image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # These two small vision models contain operations that keep float32
    # intermediates; using BF16 weights causes mixed bias/input failures in the
    # current Transformers kernels on GB10.
    dtype = torch.float32
    detector = segmenter = detector_processor = segmenter_processor = None
    try:
        detector_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny", torch_dtype=dtype
        ).to(device).eval()
        detector_inputs = detector_processor(images=image, text=request.prompt, return_tensors="pt").to(device)
        detector_inputs["pixel_values"] = detector_inputs["pixel_values"].to(dtype=dtype)
        with torch.inference_mode():
            detector_outputs = detector(**detector_inputs)
        detections = detector_processor.post_process_grounded_object_detection(
            detector_outputs,
            detector_inputs.input_ids,
            threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = detections["boxes"].detach().float().cpu()
        if boxes.numel() == 0:
            raise ValueError(f"no region matched mask prompt: {request.prompt}")

        # Release the detector before loading SAM so peak unified-memory use
        # remains bounded while the Krea pipeline is resident.
        del detector_outputs, detector_inputs, detector
        detector = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

        segmenter_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
        segmenter = Sam2Model.from_pretrained(
            "facebook/sam2.1-hiera-small", torch_dtype=dtype
        ).to(device).eval()
        sam_inputs = segmenter_processor(
            images=image,
            input_boxes=[boxes.tolist()],
            return_tensors="pt",
        ).to(device)
        sam_inputs["pixel_values"] = sam_inputs["pixel_values"].to(dtype=dtype)
        with torch.inference_mode():
            sam_outputs = segmenter(**sam_inputs, multimask_output=False)
        masks = segmenter_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu()
        )[0]
        union = (masks.squeeze(1) > request.mask_threshold).any(dim=0).to(torch.uint8).numpy() * 255
        mask = Image.fromarray(union, mode="L")
        if request.grow:
            kernel = max(3, request.grow * 2 + 1)
            if kernel % 2 == 0:
                kernel += 1
            mask = mask.filter(ImageFilter.MaxFilter(kernel))
        if request.feather:
            mask = mask.filter(ImageFilter.GaussianBlur(request.feather))
        output = io.BytesIO()
        mask.save(output, format="PNG")
        return {
            "mask_b64_json": base64.b64encode(output.getvalue()).decode("ascii"),
            "boxes": boxes.tolist(),
            "scores": detections["scores"].detach().float().cpu().tolist(),
        }
    finally:
        del detector, segmenter, detector_processor, segmenter_processor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass


@app.post("/v1/masks/segment")
async def segment(request: SegmentRequest) -> dict[str, Any]:
    prompt = request.prompt.strip().rstrip(".")
    if not prompt:
        raise HTTPException(status_code=400, detail="mask prompt is required")
    if not 0 <= request.box_threshold <= 1 or not 0 <= request.text_threshold <= 1 or not 0 <= request.mask_threshold <= 1:
        raise HTTPException(status_code=400, detail="mask thresholds must be between 0 and 1")
    if request.grow < 0 or request.grow > 64 or request.feather < 0 or request.feather > 64:
        raise HTTPException(status_code=400, detail="mask grow and feather must be between 0 and 64")
    request.prompt = prompt
    try:
        async with segmentation_lock:
            async with generation_lock:
                return await asyncio.to_thread(_automatic_mask, request)
    except (ValueError, RuntimeError, UnidentifiedImageError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/images/generations")
async def generate(request: ImageRequest) -> dict[str, Any]:
    global runtime_profile, runtime_signature, runtime_stage
    global runtime_started_at, runtime_last_load_seconds, runtime_error
    if request.model not in MODEL_ALIASES:
        raise HTTPException(status_code=400, detail=f"model mismatch: {request.model}")
    if request.n != 1:
        raise HTTPException(status_code=400, detail="only n=1 is supported")
    if request.response_format != "b64_json":
        raise HTTPException(status_code=400, detail="only b64_json is supported")
    if request.checkpoint not in CHECKPOINT_MODELS:
        raise HTTPException(status_code=400, detail=f"unsupported checkpoint: {request.checkpoint}")
    if request.checkpoint.endswith("-nvfp4"):
        source_key = request.checkpoint.removesuffix("-nvfp4")
        if not nvfp4_validated(source_key):
            raise HTTPException(status_code=409, detail=f"converted checkpoint is not validated: {request.checkpoint}")
    elif request.checkpoint not in OFFICIAL_CHECKPOINTS and not ready(CHECKPOINTS[request.checkpoint]):
        raise HTTPException(status_code=409, detail=f"checkpoint is not prepared: {request.checkpoint}")
    if request.checkpoint not in OFFICIAL_CHECKPOINTS and request.filter_mode != "off":
        raise HTTPException(status_code=400, detail="third-party checkpoints already include tuning; set filter_mode=off")
    if request.reid_image and request.checkpoint not in OFFICIAL_CHECKPOINTS:
        raise HTTPException(status_code=400, detail="Krea ReID currently requires an official checkpoint")
    if request.character_sheet_image and request.checkpoint not in OFFICIAL_CHECKPOINTS:
        raise HTTPException(status_code=400, detail="CharacterSheet currently requires an official checkpoint")
    diffusion_model = CHECKPOINT_MODELS[request.checkpoint]
    if request.sampler_name not in {None, "euler", "euler_ancestral", "er_sde"}:
        raise HTTPException(status_code=400, detail="sampler_name must be euler, euler_ancestral, or er_sde")
    if request.scheduler not in {None, "simple", "beta"}:
        raise HTTPException(status_code=400, detail="scheduler must be simple or beta")
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if request.identity_preset == "headSwap":
        # Preserve the functional LoRA's trained trigger verbatim.  Generic
        # natural-language rewrites were the reason the old Identity-only path
        # often kept the target hair or copied unrelated reference details.
        prompt = "head_swap: replace the head with the reference head."
    width, height = parse_size(request.size)
    if not 0 <= request.control_strength <= 2:
        raise HTTPException(status_code=400, detail="control_strength must be between 0 and 2")
    if not 0 <= request.identity_strength <= 2:
        raise HTTPException(status_code=400, detail="identity_strength must be between 0 and 2")
    if not 0 <= request.ref_boost <= 20:
        raise HTTPException(status_code=400, detail="ref_boost must be between 0 and 20")
    if not 0 <= request.source_ref_boost <= 20:
        raise HTTPException(status_code=400, detail="source_ref_boost must be between 0 and 20")
    reference_images = ([request.reference_image] if request.reference_image else []) + list(request.reference_images)
    if len(reference_images) > 3:
        raise HTTPException(status_code=400, detail="at most three identity reference images are supported")
    if reference_images and not request.source_image:
        raise HTTPException(status_code=400, detail="identity references require source_image")
    if request.reid_image and (
        request.source_image
        or request.control_image
        or request.vision_images
        or request.style_reference_images
        or request.nk2e_image
        or request.anypaint_image
        or request.detail_enhance_image
    ):
        raise HTTPException(status_code=400, detail="Krea ReID cannot be combined with other image-reference modules")
    if request.character_sheet_image and (
        request.reid_image
        or request.source_image
        or request.control_image
        or request.vision_images
        or request.style_reference_images
        or request.nk2e_image
        or request.anypaint_image
        or request.detail_enhance_image
        or request.style is not None
        or request.styles
        or request.user_loras
    ):
        raise HTTPException(status_code=400, detail="CharacterSheet cannot be combined with other modules")
    if not 384 <= request.grounding_px <= 1024:
        raise HTTPException(status_code=400, detail="grounding_px must be between 384 and 1024")
    if request.style not in {None, *STYLE_LORAS}:
        raise HTTPException(status_code=400, detail=f"unsupported style: {request.style}")
    if not 0 <= request.style_strength <= 2:
        raise HTTPException(status_code=400, detail="style_strength must be between 0 and 2")
    styles = list(request.styles)
    if request.style is not None and not any(style.name == request.style for style in styles):
        styles.append(StyleSelection(name=request.style, strength=request.style_strength))
    if len(styles) > len(STYLE_LORAS):
        raise HTTPException(status_code=400, detail="too many style LoRAs")
    if len({style.name for style in styles}) != len(styles):
        raise HTTPException(status_code=400, detail="duplicate style LoRAs are not supported")
    for style in styles:
        if style.name not in STYLE_LORAS:
            raise HTTPException(status_code=400, detail=f"unsupported style: {style.name}")
        if not 0 <= style.strength <= 2:
            raise HTTPException(status_code=400, detail="style strength must be between 0 and 2")
    user_loras = list(request.user_loras)
    if len(user_loras) > 5:
        raise HTTPException(status_code=400, detail="at most five user LoRAs may be stacked")
    if len({selection.filename for selection in user_loras}) != len(user_loras):
        raise HTTPException(status_code=400, detail="duplicate user LoRAs are not supported")
    for selection in user_loras:
        if selection.filename == "skc3vo.safetensors":
            raise HTTPException(status_code=400, detail="skc3vo is a filter mode; use filter_mode=adherence")
        filename = Path(selection.filename).name
        if filename != selection.filename or not filename.endswith(".safetensors"):
            raise HTTPException(status_code=400, detail="invalid user LoRA filename")
        if not (USER_LORA_ROOT / filename).is_file():
            raise HTTPException(status_code=400, detail=f"user LoRA not found: {filename}")
        if not -2 <= selection.strength <= 2:
            raise HTTPException(status_code=400, detail="user LoRA strength must be between -2 and 2")
    if len(request.vision_images) > 4:
        raise HTTPException(status_code=400, detail="at most four vision reference images are supported")
    if len(request.style_reference_images) > 2:
        raise HTTPException(status_code=400, detail="at most two style reference images are supported")
    if request.vision_mode not in {"descriptor", "instruct"}:
        raise HTTPException(status_code=400, detail="vision_mode must be descriptor or instruct")
    if not 0.1 <= request.vision_megapixels <= 4:
        raise HTTPException(status_code=400, detail="vision_megapixels must be between 0.1 and 4")
    if request.vision_images and request.source_image:
        raise HTTPException(status_code=400, detail="vision references cannot yet be combined with identity edit")
    if not 0 <= request.style_reference_strength <= 2:
        raise HTTPException(status_code=400, detail="style_reference_strength must be between 0 and 2")
    if request.style_reference_images and (
        request.vision_images or request.source_image or request.control_image or styles or user_loras
    ):
        raise HTTPException(
            status_code=400,
            detail="style references cannot yet be combined with vision, identity, depth, or style presets",
        )
    if request.style_reference_images and request.checkpoint not in OFFICIAL_CHECKPOINTS:
        raise HTTPException(
            status_code=400,
            detail="style reference currently uses its own fixed official INT8 checkpoint; select an official checkpoint",
        )
    if request.nk2e_mode not in {"edit", "canny"}:
        raise HTTPException(status_code=400, detail="nk2e_mode must be edit or canny")
    if not 0 <= request.nk2e_strength <= 2:
        raise HTTPException(status_code=400, detail="nk2e_strength must be between 0 and 2")
    if (request.identity_mask or request.strict_mask) and not request.source_image:
        raise HTTPException(status_code=400, detail="identity masks require source_image")
    if not 0 <= request.strict_mask_grow <= 128 or not 0 <= request.strict_mask_feather <= 128:
        raise HTTPException(status_code=400, detail="strict mask grow and feather must be between 0 and 128")
    if request.vae_mode not in {"default", "wan", "real"}:
        raise HTTPException(status_code=400, detail="vae_mode must be default, wan, or real")
    if request.identity_fit_mode not in {"fit", "crop"}:
        raise HTTPException(status_code=400, detail="identity_fit_mode must be fit or crop")
    if request.identity_model not in {"selected", "convrot"}:
        raise HTTPException(status_code=400, detail="identity_model must be selected or convrot")
    if request.identity_encoder not in {"default", "heretic"}:
        raise HTTPException(status_code=400, detail="identity_encoder must be default or heretic")
    if request.identity_preset not in {"", "restage", "sheet", "faceSwap", "headSwap", "personSwap", "tryon", "replace"}:
        raise HTTPException(status_code=400, detail="unsupported identity_preset")
    if request.filter_mode not in {"off", "adherence", "balanced", "strong"}:
        raise HTTPException(status_code=400, detail="filter_mode must be off, adherence, balanced, or strong")
    if request.filter_strength is not None and not 0 <= request.filter_strength <= 10:
        raise HTTPException(status_code=400, detail="filter_strength must be between 0 and 10")
    if not 0 <= request.prompt_enhancer_strength <= 2:
        raise HTTPException(status_code=400, detail="prompt_enhancer_strength must be between 0 and 2")
    if not 0.25 <= request.prompt_text_scale <= 4:
        raise HTTPException(status_code=400, detail="prompt_text_scale must be between 0.25 and 4")
    if not 0 <= request.detail_strength <= 2:
        raise HTTPException(status_code=400, detail="detail_strength must be between 0 and 2")
    if request.detail_vae not in {"wan", "qwen"}:
        raise HTTPException(status_code=400, detail="detail_vae must be wan or qwen")
    if request.nk2e_image and (
        request.style_reference_images
        or request.vision_images
        or request.source_image
        or request.control_image
        or styles
        or user_loras
    ):
        raise HTTPException(status_code=400, detail="NK2E experiments cannot be combined with other Krea modules")
    if request.anypaint_mask and not request.anypaint_image:
        raise HTTPException(status_code=400, detail="anypaint_mask requires anypaint_image")
    if request.anypaint_image and (
        request.style_reference_images
        or request.vision_images
        or request.source_image
        or reference_images
        or request.control_image
        or styles
        or user_loras
        or request.nk2e_image
    ):
        raise HTTPException(status_code=400, detail="AnyPaint cannot be combined with other Krea modules")
    if request.detail_enhance_image and (
        request.style_reference_images
        or request.vision_images
        or request.source_image
        or reference_images
        or request.control_image
        or styles
        or user_loras
        or request.nk2e_image
        or request.anypaint_image
    ):
        raise HTTPException(status_code=400, detail="detail enhancement cannot be combined with other Krea modules")
    pads = (
        request.outpaint_left,
        request.outpaint_top,
        request.outpaint_right,
        request.outpaint_bottom,
    )
    if any(value < 0 or value > 1536 or value % 16 for value in pads):
        raise HTTPException(status_code=400, detail="outpaint padding must be 0..1536 in multiples of 16")
    if not 0 <= request.anypaint_strength <= 2:
        raise HTTPException(status_code=400, detail="anypaint_strength must be between 0 and 2")
    if not 128 <= request.anypaint_reference_max_edge <= 768 or request.anypaint_reference_max_edge % 16:
        raise HTTPException(status_code=400, detail="anypaint_reference_max_edge must be 128..768 in multiples of 16")
    if not 0 <= request.anypaint_boundary_redraw_px <= 256:
        raise HTTPException(status_code=400, detail="anypaint_boundary_redraw_px must be between 0 and 256")
    steps = request.steps if request.steps is not None else (
        10 if request.source_image or request.detail_enhance_image else 8
    )
    if not 1 <= steps <= 20:
        raise HTTPException(status_code=400, detail="steps must be between 1 and 20")
    # Keep random seeds within JavaScript's exact integer range so the web client
    # can clone and reproduce them without rounding.
    seed = request.seed if request.seed is not None and request.seed >= 0 else secrets.randbits(53)
    for style in styles:
        trigger = STYLE_TRIGGERS[style.name]
        if trigger.lower() not in prompt.lower():
            prompt = f"{prompt}. {trigger}"
    prefix = f"nvfp4-api/{uuid.uuid4().hex}"
    depth_path: Path | None = None
    depth_preview: str | None = None
    source_path: Path | None = None
    reference_paths: list[Path] = []
    prepared_depth_path: Path | None = None
    identity_mask_path: Path | None = None
    strict_mask_path: Path | None = None
    vision_paths: list[Path] = []
    style_reference_paths: list[Path] = []
    nk2e_path: Path | None = None
    nk2e_preview: str | None = None
    anypaint_path: Path | None = None
    anypaint_mask_path: Path | None = None
    detail_path: Path | None = None

    target_profile = request_runtime_profile(request)
    target_signature = request_runtime_signature(request)
    if request.prepare_only and runtime_stage == "ready" and runtime_signature == target_signature:
        return {
            "created": int(time.time()),
            "prepared": True,
            "warm": True,
            "profile": runtime_profile,
            "signature": runtime_signature,
            "load_seconds": 0,
        }
    async with generation_lock:
        runtime_stage = "preparing" if request.prepare_only else "generating"
        runtime_started_at = time.monotonic()
        runtime_error = ""
        publish_runtime_operation(
            request.operation_id,
            "preparing",
            target_profile,
            "Krea 실행 그래프와 입력 준비",
            0.04,
        )
        try:
            depth_name: str | None = None
            # A pose image used together with Identity Edit is a semantic source-B
            # reference, exactly like the published Krea2 Edit workflow.  Only a
            # standalone structure-control request is converted to monocular depth.
            if request.control_image and not request.source_image:
                depth_name, depth_path, depth_preview = await asyncio.to_thread(
                    save_depth_input, request.control_image
                )
            vision_names: list[str] = []
            for encoded_image in request.vision_images:
                vision_name, vision_path = await asyncio.to_thread(
                    save_input, encoded_image, "krea-vision"
                )
                vision_names.append(vision_name)
                vision_paths.append(vision_path)
            style_reference_names: list[str] = []
            for encoded_image in request.style_reference_images:
                image_name, image_path = await asyncio.to_thread(
                    save_input, encoded_image, "krea-style-reference"
                )
                style_reference_names.append(image_name)
                style_reference_paths.append(image_path)
            nk2e_name: str | None = None
            if request.nk2e_image:
                nk2e_name, nk2e_path, nk2e_preview = await asyncio.to_thread(
                    save_nk2e_input, request.nk2e_image, request.nk2e_mode, request.nk2e_preprocessed
                )
            anypaint_name: str | None = None
            anypaint_mask_name: str | None = None
            if request.anypaint_image:
                anypaint_name, anypaint_path = await asyncio.to_thread(
                    save_input, request.anypaint_image, "krea-anypaint"
                )
                source_width, source_height = Image.open(anypaint_path).size
                output_width = source_width + request.outpaint_left + request.outpaint_right
                output_height = source_height + request.outpaint_top + request.outpaint_bottom
                if not (512 <= output_width <= 2048 and 512 <= output_height <= 2048):
                    raise HTTPException(status_code=400, detail="AnyPaint output dimensions must be between 512 and 2048")
                if output_width % 16 or output_height % 16:
                    raise HTTPException(status_code=400, detail="AnyPaint source plus padding must be multiples of 16")
                if request.anypaint_mask:
                    anypaint_mask_name, anypaint_mask_path = await asyncio.to_thread(
                        save_input, request.anypaint_mask, "krea-anypaint-mask"
                    )
                    if Image.open(anypaint_mask_path).size != (source_width, source_height):
                        raise HTTPException(status_code=400, detail="AnyPaint mask dimensions must match the source image")
            detail_name: str | None = None
            if request.detail_enhance_image:
                detail_name, detail_path = await asyncio.to_thread(
                    save_input, request.detail_enhance_image, "krea-detail-enhance"
                )
                width, height = Image.open(detail_path).size
                if not (512 <= width <= 2048 and 512 <= height <= 2048):
                    raise HTTPException(status_code=400, detail="detail image dimensions must be between 512 and 2048")
                if width % 16 or height % 16:
                    raise HTTPException(status_code=400, detail="detail image dimensions must be multiples of 16")
            if detail_name is not None:
                graph = detail_enhance_workflow(
                    prompt,
                    width,
                    height,
                    seed,
                    prefix,
                    detail_name,
                    request.detail_strength,
                    steps,
                    WAN_VAE if request.detail_vae == "wan" else VAE,
                    diffusion_model,
                )
            elif anypaint_name is not None:
                graph = anypaint_workflow(
                    prompt,
                    seed,
                    prefix,
                    anypaint_name,
                    anypaint_mask_name,
                    request.outpaint_left,
                    request.outpaint_top,
                    request.outpaint_right,
                    request.outpaint_bottom,
                    request.anypaint_strength,
                    request.anypaint_reference_max_edge,
                    request.anypaint_boundary_redraw_px,
                    request.anypaint_vlm_reference,
                    steps,
                    diffusion_model,
                )
            elif nk2e_name is not None:
                graph = nk2e_workflow(
                    prompt,
                    width,
                    height,
                    seed,
                    prefix,
                    nk2e_name,
                    request.nk2e_mode,
                    request.nk2e_strength,
                    steps,
                    diffusion_model,
                )
            elif style_reference_names:
                graph = style_reference_workflow(
                    prompt,
                    width,
                    height,
                    seed,
                    prefix,
                    style_reference_names,
                    request.style_reference_strength,
                    steps,
                )
            elif request.character_sheet_image:
                source_name, source_path = await asyncio.to_thread(save_input, request.character_sheet_image, "krea-character-sheet")
                graph = character_sheet_workflow(source_name, seed, prefix, steps)
            elif request.reid_image:
                source_name, source_path = await asyncio.to_thread(
                    save_input, request.reid_image, "krea-reid"
                )
                graph = reid_workflow(
                    prompt,
                    width,
                    height,
                    seed,
                    prefix,
                    source_name,
                    steps,
                    styles,
                    user_loras,
                )
            elif request.source_image:
                source_name, source_path = await asyncio.to_thread(
                    save_input, request.source_image, "krea-edit"
                )
                identity_mask_name: str | None = None
                if request.identity_mask:
                    identity_mask_name, identity_mask_path = await asyncio.to_thread(
                        save_input, request.identity_mask, "krea-edit-mask"
                    )
                    if Image.open(identity_mask_path).size != Image.open(source_path).size:
                        raise HTTPException(status_code=400, detail="identity mask dimensions must match source image")
                if request.strict_mask:
                    _, strict_mask_path = await asyncio.to_thread(
                        save_input, request.strict_mask, "krea-strict-mask"
                    )
                    if Image.open(strict_mask_path).size != Image.open(source_path).size:
                        raise HTTPException(status_code=400, detail="strict mask dimensions must match source image")
                reference_names: list[str] = []
                identity_references = list(reference_images)
                # The official/community Identity Edit recipe stitches clothing,
                # pose and prop references horizontally and feeds that single image
                # to source B.  Do not replace this with Depth ControlNet: doing so
                # loses the semantic role of the pose image and makes the two LoRAs
                # compete with each other.
                if request.control_image:
                    identity_references.append(request.control_image)
                for encoded_reference in identity_references:
                    reference_name, reference_path = await asyncio.to_thread(
                        save_identity_reference, encoded_reference
                    )
                    reference_names.append(reference_name)
                    reference_paths.append(reference_path)
                identity_diffusion_model = (
                    HEAD_SWAP_MODEL
                    if request.identity_preset == "headSwap"
                    else IDENTITY_EDIT_MODEL
                    if request.identity_model == "convrot"
                    else diffusion_model
                )
                # Identity Edit's demonstrated conditioning path uses the
                # Heretic ConvRot encoder. Keep it for compatible third-party
                # Krea checkpoints too; otherwise changing only the checkpoint
                # also silently changes the instruction encoder.
                identity_text_encoder = (
                    TEXT_ENCODER
                    if request.identity_preset == "headSwap"
                    else IDENTITY_EDIT_TEXT_ENCODER
                    if request.identity_encoder == "heretic"
                    else TEXT_ENCODER
                )
                graph = identity_workflow(
                    prompt,
                    width,
                    height,
                    seed,
                    prefix,
                    source_name,
                    reference_names,
                    identity_mask_name,
                    request.identity_strength,
                    request.ref_boost,
                    request.source_ref_boost,
                    request.grounding_px,
                    steps,
                    styles,
                    user_loras,
                    None,
                    request.control_strength,
                    request.identity_fit_mode,
                    identity_diffusion_model,
                    identity_text_encoder,
                    request.checkpoint != "chriscole-edit-v1.1",
                    request.identity_preset,
                )
            elif depth_name is not None:
                graph = depth_workflow(
                    prompt,
                    width,
                    height,
                    seed,
                    prefix,
                    depth_name,
                    request.control_strength,
                    steps,
                    styles,
                    user_loras,
                    diffusion_model,
                )
                if vision_names:
                    graph = apply_vision_conditioning(
                        graph,
                        prompt,
                        vision_names,
                        request.vision_mode,
                        request.vision_megapixels,
                    )
            else:
                graph = workflow(prompt, width, height, seed, prefix, steps, styles, user_loras, diffusion_model)
                if vision_names:
                    graph = apply_vision_conditioning(
                        graph,
                        prompt,
                        vision_names,
                        request.vision_mode,
                        request.vision_megapixels,
                    )
            if not request.reid_image and not request.character_sheet_image:
                # User LoRAs and filter-bypass LoRAs modify the same DiT
                # weights. Stacking them weakened trained facial traits in
                # regression tests, so user LoRAs always take precedence.
                if not user_loras:
                    graph = apply_filter_bypass(graph, request.filter_mode, request.filter_strength)
                graph = apply_prompt_enhancer(
                    graph,
                    request.prompt_enhancer,
                    request.prompt_enhancer_strength,
                    request.prompt_text_scale,
                )
            recommended_sampler, recommended_scheduler = CHECKPOINT_SAMPLING.get(
                request.checkpoint,
                ("euler", "simple"),
            )
            sampler_name = "euler" if request.reid_image or request.character_sheet_image else request.sampler_name or (
                "er_sde" if detail_name is not None else recommended_sampler
            )
            scheduler = "simple" if request.reid_image or request.character_sheet_image else request.scheduler or recommended_scheduler
            graph = apply_sampling(graph, sampler_name, scheduler)
            if "3" in graph and detail_name is None and not request.reid_image and not request.character_sheet_image:
                if request.vae_mode == "real":
                    graph["3"]["inputs"]["vae_name"] = REAL_VAE
                elif request.vae_mode == "wan":
                    graph["3"]["inputs"]["vae_name"] = WAN_VAE
            publish_runtime_operation(
                request.operation_id,
                "model_loading",
                target_profile,
                "체크포인트·인코더·VAE·LoRA 상태 확인 및 탑재",
                0.1,
                "load",
            )
            encoded = await execute_workflow(graph, request.operation_id)
            runtime_profile = target_profile
            runtime_signature = target_signature
            runtime_last_load_seconds = time.monotonic() - runtime_started_at
            runtime_stage = "ready"
            publish_runtime_operation(
                request.operation_id,
                "cache_retaining",
                target_profile,
                "ComfyUI 모델 캐시 유지",
                0.98,
                "retain",
                True,
            )
            if strict_mask_path is not None and source_path is not None:
                encoded = await asyncio.to_thread(
                    composite_strict_mask,
                    encoded,
                    source_path,
                    strict_mask_path,
                    request.strict_mask_grow,
                    request.strict_mask_feather,
                )
        except (httpx.HTTPError, KeyError, RuntimeError, TimeoutError) as exc:
            runtime_error = str(exc)
            runtime_stage = "error"
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if depth_path is not None:
                depth_path.unlink(missing_ok=True)
            if source_path is not None:
                source_path.unlink(missing_ok=True)
            for reference_path in reference_paths:
                reference_path.unlink(missing_ok=True)
            if prepared_depth_path is not None:
                prepared_depth_path.unlink(missing_ok=True)
            if identity_mask_path is not None:
                identity_mask_path.unlink(missing_ok=True)
            if strict_mask_path is not None:
                strict_mask_path.unlink(missing_ok=True)
            for vision_path in vision_paths:
                vision_path.unlink(missing_ok=True)
            for style_reference_path in style_reference_paths:
                style_reference_path.unlink(missing_ok=True)
            if nk2e_path is not None:
                nk2e_path.unlink(missing_ok=True)
            if anypaint_path is not None:
                anypaint_path.unlink(missing_ok=True)
            if anypaint_mask_path is not None:
                anypaint_mask_path.unlink(missing_ok=True)
            if detail_path is not None:
                detail_path.unlink(missing_ok=True)

    if request.prepare_only:
        publish_runtime_operation(
            request.operation_id,
            "completed",
            target_profile,
            "요청한 Krea 런타임 준비 완료",
            1.0,
            "retain",
            True,
        )
        return {
            "created": int(time.time()),
            "prepared": True,
            "profile": runtime_profile,
            "signature": runtime_signature,
            "load_seconds": runtime_last_load_seconds,
        }
    publish_runtime_operation(
        request.operation_id,
        "completed",
        target_profile,
        "이미지 생성과 캐시 정리 완료",
        1.0,
        "retain",
        True,
    )
    response = {"created": int(time.time()), "seed": seed, "data": [{"b64_json": encoded}]}
    if depth_preview is not None:
        response["control_b64_json"] = depth_preview
    if nk2e_preview is not None:
        response["control_b64_json"] = nk2e_preview
    return response
