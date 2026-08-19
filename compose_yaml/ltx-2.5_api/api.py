import asyncio
import logging
import os
import secrets
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ltx-api")

MODEL_DIR = Path(os.environ.get("LTX_MODEL_DIR", "/models/ltx-2.5"))
TRANSFORMER = MODEL_DIR / "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
TEXT_ENCODER = MODEL_DIR / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE = MODEL_DIR / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO_VAE = MODEL_DIR / "vae/ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER = MODEL_DIR / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

app = FastAPI(title="LTX-2.5 NVFP4 API", version="1")
load_lock = asyncio.Lock()
generation_lock = asyncio.Lock()
pipeline: DistilledPipeline | None = None
load_error = ""


def _paths() -> ModelPaths:
    return ModelPaths.from_split(
        transformer_path=str(TRANSFORMER),
        text_encoder_path=str(TEXT_ENCODER),
        video_vae_path=str(VIDEO_VAE),
        audio_vae_path=str(AUDIO_VAE),
    )


def _load_pipeline() -> DistilledPipeline:
    log.info("loading LTX-2.5 distilled NVFP4 pipeline")
    policy = QuantizationKind.NVFP4_PREQUANT.to_policy(str(TRANSFORMER))
    loaded = DistilledPipeline(
        model_paths=_paths(),
        spatial_upsampler_path=str(UPSCALER),
        loras=(),
        quantization=policy,
        offload_mode=OffloadMode.NONE,
    )
    log.info("LTX-2.5 pipeline ready")
    return loaded


async def get_pipeline() -> DistilledPipeline:
    global pipeline, load_error
    if pipeline is not None:
        return pipeline
    async with load_lock:
        if pipeline is not None:
            return pipeline
        try:
            pipeline = await asyncio.to_thread(_load_pipeline)
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
        "loaded": pipeline is not None,
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
    image_path: str | None,
    image_strength: float,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    seed: int,
) -> None:
    images = []
    if image_path:
        images.append(ImageConditioningInput(path=image_path, frame_idx=0, strength=image_strength))
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
    image_strength: float = Form(1.0),
    image: UploadFile | None = File(None),
) -> FileResponse:
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")
    _validate_dimensions(width, height, num_frames, fps)
    if not 0 <= image_strength <= 1:
        raise HTTPException(400, "image_strength must be between 0 and 1")
    if generation_lock.locked():
        raise HTTPException(409, "another video generation is running")
    if seed < 0:
        seed = secrets.randbelow(2**31)

    temp_dir = tempfile.TemporaryDirectory(prefix="ltx-api-")
    root = Path(temp_dir.name)
    image_path = None
    if image is not None:
        suffix = Path(image.filename or "image.png").suffix.lower() or ".png"
        image_path = root / f"input{suffix}"
        data = await image.read()
        if len(data) > 32 << 20:
            temp_dir.cleanup()
            raise HTTPException(413, "image is too large (max 32 MB)")
        image_path.write_bytes(data)

    output = root / "output.mp4"
    try:
        async with generation_lock:
            loaded = await get_pipeline()
            await asyncio.to_thread(
                _generate,
                loaded,
                prompt,
                str(output),
                str(image_path) if image_path else None,
                image_strength,
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
