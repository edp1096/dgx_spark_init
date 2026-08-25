import asyncio
import gc
import json
import logging
import os
import secrets
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode
from prepare_models import BASE_FILES, PUBLIC_FILES, missing_paths, prepare_all


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ltx-api")

MODEL_DIR = Path(os.environ.get("LTX_MODEL_DIR", "/models/ltx-2.5"))
TRANSFORMER = MODEL_DIR / "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
TEXT_ENCODER = Path(
    os.environ.get(
        "LTX_TEXT_ENCODER_PATH",
        str(MODEL_DIR / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
    )
)
VIDEO_VAE = MODEL_DIR / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO_VAE = MODEL_DIR / "vae/ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER = MODEL_DIR / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
LORA_PATH = os.environ.get("LTX_LORA_PATH", "").strip()
LORA_STRENGTH = float(os.environ.get("LTX_LORA_STRENGTH", "1.0"))
TOKEN_FILE = Path("/root/.cache/huggingface/media-secrets/hf_token")
MOTION_LORA_PATH = MODEL_DIR / "loras/ltx-2.3-ltx2-better-nsfw-motion.safetensors"

app = FastAPI(title="LTX-2.5 NVFP4 API", version="1")
load_lock = asyncio.Lock()
generation_lock = asyncio.Lock()
prepare_lock = asyncio.Lock()
pipeline: DistilledPipeline | None = None
pipeline_signature: tuple[str, float] | None = None
load_error = ""
prepare_error = ""
preparing = False
prepare_task: asyncio.Task[None] | None = None


class ModelPrepareRequest(BaseModel):
    hf_token: str = ""


def _stored_token() -> str:
    try:
        return TOKEN_FILE.read_text().strip()
    except OSError:
        return ""


def _save_token(token: str) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)


def _token_configured() -> bool:
    return bool(_stored_token() or os.environ.get("HF_TOKEN", "").strip())


def _model_status() -> dict[str, object]:
    missing = missing_paths(MODEL_DIR)
    return {
        "ready": not missing,
        "preparing": preparing,
        "token_configured": _token_configured(),
        "missing": [str(path.relative_to(MODEL_DIR)) for path in missing],
        "required_files": len(BASE_FILES) + len(PUBLIC_FILES),
        "ready_files": len(BASE_FILES) + len(PUBLIC_FILES) - len(missing),
        "motion_lora_ready": not any(str(path).startswith(str(MODEL_DIR / "loras")) for path in missing),
        "error": prepare_error,
    }


async def _prepare(token: str) -> None:
    global prepare_error, preparing
    async with prepare_lock:
        preparing = True
        prepare_error = ""
        try:
            await asyncio.to_thread(prepare_all, MODEL_DIR, token or None)
        except BaseException as exc:
            prepare_error = str(exc)
            log.exception("model preparation failed")
        finally:
            preparing = False


def _start_prepare(token: str = "") -> bool:
    global prepare_task
    if prepare_task is not None and not prepare_task.done():
        return False
    effective_token = token or _stored_token() or os.environ.get("HF_TOKEN", "").strip()
    prepare_task = asyncio.create_task(_prepare(effective_token))
    return True


@app.on_event("startup")
async def prepare_missing_models() -> None:
    if missing_paths(MODEL_DIR):
        _start_prepare()


@app.get("/v1/models/status")
def model_status() -> dict[str, object]:
    return _model_status()


@app.post("/v1/models/prepare", status_code=202)
async def prepare_models(request: ModelPrepareRequest) -> dict[str, object]:
    token = request.hf_token.strip()
    if token:
        if not token.startswith("hf_") or len(token) < 12:
            raise HTTPException(400, "Hugging Face read token must start with hf_")
        _save_token(token)
    if not _token_configured() and any(not (MODEL_DIR / name).is_file() for name in BASE_FILES):
        raise HTTPException(400, "Accept the LTX-2.5 license and enter a Hugging Face read token")
    started = _start_prepare(token)
    return {**_model_status(), "started": started}


def _paths() -> ModelPaths:
    return ModelPaths.from_split(
        transformer_path=str(TRANSFORMER),
        text_encoder_path=str(TEXT_ENCODER),
        video_vae_path=str(VIDEO_VAE),
        audio_vae_path=str(AUDIO_VAE),
    )


def _load_pipeline(lora_path: str, lora_strength: float) -> DistilledPipeline:
    log.info("loading LTX-2.5 distilled NVFP4 pipeline")
    policy = QuantizationKind.NVFP4_PREQUANT.to_policy(str(TRANSFORMER))
    loras: list[LoraPathStrengthAndSDOps] = []
    if lora_path:
        if not Path(lora_path).is_file():
            raise FileNotFoundError(f"LTX LoRA not found: {lora_path}")
        log.info("fusing LoRA %s at strength %.3f", lora_path, lora_strength)
        loras.append(
            LoraPathStrengthAndSDOps(
                lora_path,
                lora_strength,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        )
    loaded = DistilledPipeline(
        model_paths=_paths(),
        spatial_upsampler_path=str(UPSCALER),
        loras=loras,
        quantization=policy,
        offload_mode=OffloadMode.NONE,
    )
    log.info("LTX-2.5 pipeline ready")
    return loaded


async def get_pipeline(lora_path: str, lora_strength: float) -> DistilledPipeline:
    global pipeline, pipeline_signature, load_error
    signature = (lora_path, round(lora_strength, 4))
    if pipeline is not None and pipeline_signature == signature:
        return pipeline
    async with load_lock:
        if pipeline is not None and pipeline_signature == signature:
            return pipeline
        try:
            if pipeline is not None:
                previous = pipeline
                pipeline = None
                pipeline_signature = None
                del previous
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            pipeline = await asyncio.to_thread(_load_pipeline, lora_path, lora_strength)
            pipeline_signature = signature
            load_error = ""
        except Exception as exc:
            load_error = str(exc)
            log.exception("pipeline load failed")
            raise
    return pipeline


@app.get("/health")
def health() -> dict[str, object]:
    missing = [str(path) for path in (TRANSFORMER, TEXT_ENCODER, VIDEO_VAE, AUDIO_VAE, UPSCALER) if not path.is_file()]
    status = "ready" if pipeline is not None else "idle"
    if load_error:
        status = "error"
    return {
        "status": status,
        "model": "Lightricks/LTX-2.5 distilled NVFP4",
        "text_encoder": str(TEXT_ENCODER),
        "loaded": pipeline is not None,
        "lora": pipeline_signature[0] if pipeline_signature and pipeline_signature[0] else None,
        "lora_strength": pipeline_signature[1] if pipeline_signature and pipeline_signature[0] else None,
        "busy": generation_lock.locked(),
        "missing": missing,
        "error": load_error,
    }


def _validate_dimensions(width: int, height: int, num_frames: int, fps: float) -> None:
    if width < 256 or height < 256 or width % 64 or height % 64:
        raise HTTPException(400, "width and height must be >= 256 and divisible by 64")
    if num_frames < 9 or (num_frames - 1) % 8:
        raise HTTPException(400, "num_frames must be 8*k+1 and at least 9")
    if not 1 <= fps <= 60:
        raise HTTPException(400, "fps must be between 1 and 60")


def _generate(
    loaded: DistilledPipeline,
    prompt: str,
    output_path: str,
    image_paths: list[str],
    frame_indices: list[int],
    image_strengths: list[float],
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    seed: int,
) -> None:
    images = [
        ImageConditioningInput(path=path, frame_idx=frame_idx, strength=strength)
        for path, frame_idx, strength in zip(image_paths, frame_indices, image_strengths, strict=True)
    ]
    with torch.inference_mode():
        video, audio, actual_frames, tiling = loaded(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            images=images,
            tiling_config=AUTO_TILING,
        )
        encode_video(
            video=video,
            fps=fps,
            audio=audio,
            output_path=output_path,
            video_chunks_number=get_video_chunks_number(actual_frames, tiling),
        )


@app.post("/v1/videos/generations")
async def generate_video(
    prompt: str = Form(...),
    width: int = Form(int(os.environ.get("LTX_DEFAULT_WIDTH", "768"))),
    height: int = Form(int(os.environ.get("LTX_DEFAULT_HEIGHT", "512"))),
    num_frames: int = Form(int(os.environ.get("LTX_DEFAULT_FRAMES", "121"))),
    fps: float = Form(float(os.environ.get("LTX_DEFAULT_FPS", "24"))),
    seed: int = Form(-1),
    frame_indices: str = Form("[]"),
    image_strengths: str = Form("[]"),
    images: list[UploadFile] | None = File(None),
    # Backward compatibility for the original single-start-image client.
    image_strength: float = Form(1.0),
    image: UploadFile | None = File(None),
    motion_lora_strength: float = Form(-1.0),
) -> FileResponse:
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")
    _validate_dimensions(width, height, num_frames, fps)
    if (motion_lora_strength < 0 and motion_lora_strength != -1) or motion_lora_strength > 1:
        raise HTTPException(400, "motion_lora_strength must be -1 or between 0 and 1")
    if motion_lora_strength < 0:
        effective_lora_path = LORA_PATH
        effective_lora_strength = LORA_STRENGTH if LORA_PATH else 0.0
    elif motion_lora_strength > 0:
        effective_lora_path = str(MOTION_LORA_PATH)
        effective_lora_strength = motion_lora_strength
    else:
        effective_lora_path = ""
        effective_lora_strength = 0.0
    try:
        parsed_indices = json.loads(frame_indices)
        parsed_strengths = json.loads(image_strengths)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "frame_indices and image_strengths must be JSON arrays") from exc
    uploads = list(images or [])
    if image is not None:
        if uploads:
            raise HTTPException(400, "use either images or the legacy image field, not both")
        uploads = [image]
        parsed_indices = [0]
        parsed_strengths = [image_strength]
    if not isinstance(parsed_indices, list) or not isinstance(parsed_strengths, list):
        raise HTTPException(400, "frame_indices and image_strengths must be JSON arrays")
    if len(uploads) != len(parsed_indices) or len(uploads) != len(parsed_strengths):
        raise HTTPException(400, "each conditioning image requires one frame index and strength")
    if len(uploads) > 10:
        raise HTTPException(400, "at most 10 conditioning images are supported")
    normalized_indices: list[int] = []
    normalized_strengths: list[float] = []
    for index, strength in zip(parsed_indices, parsed_strengths, strict=True):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= num_frames:
            raise HTTPException(400, f"conditioning frame index must be between 0 and {num_frames - 1}")
        try:
            normalized_strength = float(strength)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "conditioning strength must be numeric") from exc
        if not 0 <= normalized_strength <= 1:
            raise HTTPException(400, "conditioning strength must be between 0 and 1")
        normalized_indices.append(index)
        normalized_strengths.append(normalized_strength)
    if len(set(normalized_indices)) != len(normalized_indices):
        raise HTTPException(400, "conditioning images cannot share a frame index")
    if generation_lock.locked():
        raise HTTPException(409, "another video generation is running")
    if seed < 0:
        seed = secrets.randbelow(2**31)

    temp_dir = tempfile.TemporaryDirectory(prefix="ltx-api-")
    root = Path(temp_dir.name)
    image_paths: list[Path] = []
    for position, upload in enumerate(uploads):
        suffix = Path(upload.filename or "image.png").suffix.lower() or ".png"
        image_path = root / f"input-{position}{suffix}"
        data = await upload.read()
        if len(data) > 32 << 20:
            temp_dir.cleanup()
            raise HTTPException(413, "image is too large (max 32 MB)")
        image_path.write_bytes(data)
        image_paths.append(image_path)

    output = root / "output.mp4"
    try:
        async with generation_lock:
            if missing_paths(MODEL_DIR):
                raise HTTPException(409, "LTX model files are not ready; open Spark Media settings and prepare the video model")
            loaded = await get_pipeline(effective_lora_path, effective_lora_strength)
            await asyncio.to_thread(
                _generate,
                loaded,
                prompt,
                str(output),
                [str(path) for path in image_paths],
                normalized_indices,
                normalized_strengths,
                width,
                height,
                num_frames,
                fps,
                seed,
            )
    except HTTPException:
        temp_dir.cleanup()
        raise
    except Exception as exc:
        temp_dir.cleanup()
        log.exception("video generation failed")
        raise HTTPException(500, str(exc)) from exc

    return FileResponse(
        output,
        media_type="video/mp4",
        filename=f"ltx2.5-{seed}.mp4",
        background=_CleanupTask(temp_dir),
        headers={"X-Seed": str(seed)},
    )


class _CleanupTask:
    def __init__(self, temp_dir: tempfile.TemporaryDirectory[str]):
        self.temp_dir = temp_dir

    async def __call__(self) -> None:
        self.temp_dir.cleanup()
