import asyncio
import gc
import json
import logging
import math
import os
import secrets
import subprocess
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode
from prepare_models import A2V_FILES, BASE_FILES, PUBLIC_FILES, a2v_missing_paths, missing_paths, prepare_all
from sol_runtime import SolRuntime, normalize_mode


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ltx-api")

MODEL_DIR = Path(os.environ.get("LTX_MODEL_DIR", "/models/ltx-2.5"))
TRANSFORMER = MODEL_DIR / "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
A2V_TRANSFORMER = MODEL_DIR / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
A2V_DISTILLED_LORA = MODEL_DIR / "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
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
sol_runtime = SolRuntime(MODEL_DIR, TRANSFORMER)

app = FastAPI(title="LTX-2.5 NVFP4 API", version="1")
load_lock = asyncio.Lock()
generation_lock = asyncio.Lock()
prepare_lock = asyncio.Lock()
pipeline: DistilledPipeline | None = None
pipeline_signature: tuple[str, float] | None = None
a2v_pipeline: A2VidPipelineTwoStage | None = None
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
    a2v_missing = a2v_missing_paths(MODEL_DIR)
    return {
        "ready": not missing,
        "preparing": preparing,
        "token_configured": _token_configured(),
        "missing": [str(path.relative_to(MODEL_DIR)) for path in missing],
        "required_files": len(BASE_FILES) + len(PUBLIC_FILES) + len(A2V_FILES),
        "ready_files": len(BASE_FILES) + len(PUBLIC_FILES) + len(A2V_FILES) - len(missing) - len(a2v_missing),
        "motion_lora_ready": not any(str(path).startswith(str(MODEL_DIR / "loras")) for path in missing),
        "a2v_ready": not a2v_missing,
        "a2v_missing": [str(path.relative_to(MODEL_DIR)) for path in a2v_missing],
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


def _a2v_paths() -> ModelPaths:
    return ModelPaths.from_split(
        transformer_path=str(A2V_TRANSFORMER),
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
    global pipeline, pipeline_signature, a2v_pipeline, load_error
    signature = (lora_path, round(lora_strength, 4))
    if pipeline is not None and pipeline_signature == signature:
        return pipeline
    async with load_lock:
        if pipeline is not None and pipeline_signature == signature:
            return pipeline
        try:
            if a2v_pipeline is not None:
                previous_a2v = a2v_pipeline
                a2v_pipeline = None
                del previous_a2v
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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


def _load_a2v_pipeline() -> A2VidPipelineTwoStage:
    log.info("loading LTX-2.5 A2V dev pipeline with FP8 cast")
    policy = QuantizationKind.FP8_CAST.to_policy(str(A2V_TRANSFORMER))
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            str(A2V_DISTILLED_LORA),
            1.0,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]
    return A2VidPipelineTwoStage(
        model_paths=_a2v_paths(),
        distilled_lora=distilled_lora,
        spatial_upsampler_path=str(UPSCALER),
        loras=[],
        quantization=policy,
        offload_mode=OffloadMode.NONE,
    )


async def get_a2v_pipeline() -> A2VidPipelineTwoStage:
    global pipeline, pipeline_signature, a2v_pipeline, load_error
    if a2v_pipeline is not None:
        return a2v_pipeline
    async with load_lock:
        if a2v_pipeline is not None:
            return a2v_pipeline
        try:
            if pipeline is not None:
                previous = pipeline
                pipeline = None
                pipeline_signature = None
                del previous
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            a2v_pipeline = await asyncio.to_thread(_load_a2v_pipeline)
            load_error = ""
        except Exception as exc:
            load_error = str(exc)
            log.exception("A2V pipeline load failed")
            raise
    return a2v_pipeline


@app.get("/health")
def health() -> dict[str, object]:
    missing = [str(path) for path in (TRANSFORMER, TEXT_ENCODER, VIDEO_VAE, AUDIO_VAE, UPSCALER) if not path.is_file()]
    status = "ready" if pipeline is not None or a2v_pipeline is not None else "idle"
    if load_error:
        status = "error"
    return {
        "status": status,
        "model": "Lightricks/LTX-2.5 distilled NVFP4",
        "text_encoder": str(TEXT_ENCODER),
        "loaded": pipeline is not None or a2v_pipeline is not None,
        "pipeline": "a2v" if a2v_pipeline is not None else "distilled" if pipeline is not None else None,
        "a2v_ready": not a2v_missing_paths(MODEL_DIR),
        "lora": pipeline_signature[0] if pipeline_signature and pipeline_signature[0] else None,
        "lora_strength": pipeline_signature[1] if pipeline_signature and pipeline_signature[0] else None,
        "busy": generation_lock.locked(),
        "acceleration": sol_runtime.status(),
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
    acceleration_mode: str,
    lora_active: bool,
) -> str:
    images = [
        ImageConditioningInput(path=path, frame_idx=frame_idx, strength=strength)
        for path, frame_idx, strength in zip(image_paths, frame_indices, image_strengths, strict=True)
    ]
    with sol_runtime.activate(
        loaded,
        width,
        height,
        num_frames,
        acceleration_mode,
        lora_active=lora_active,
    ) as acceleration:
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
    return acceleration.label


def _generate_a2v(
    loaded: A2VidPipelineTwoStage,
    prompt: str,
    output_path: str,
    audio_path: str,
    audio_max_duration: float,
    image_paths: list[str],
    frame_indices: list[int],
    image_strengths: list[float],
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    seed: int,
) -> str:
    images = [
        ImageConditioningInput(path=path, frame_idx=frame_idx, strength=strength)
        for path, frame_idx, strength in zip(image_paths, frame_indices, image_strengths, strict=True)
    ]
    with torch.inference_mode():
        video, audio, tiling = loaded(
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, distorted face, malformed anatomy",
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            num_inference_steps=30,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=3.0,
                stg_scale=1.0,
                stg_blocks=[28],
                rescale_scale=0.7,
                modality_scale=3.0,
            ),
            images=images,
            audio_path=audio_path,
            audio_start_time=0.0,
            audio_max_duration=audio_max_duration,
            tiling_config=AUTO_TILING,
            max_batch_size=1,
        )
        encode_video(
            video=video,
            fps=fps,
            audio=audio,
            output_path=output_path,
            video_chunks_number=get_video_chunks_number(num_frames, tiling),
        )
    return "A2V · dev FP8"


def _normalize_a2v_audio(
    sources: list[Path], start_times: list[float], destination: Path, target_duration: float
) -> None:
    """Decode A2V input to the exact duration expected by the audio latent grid.

    LTX derives the target audio latent length from the requested video length.
    Its pipeline trims an overlong waveform but does not pad a short one, which
    otherwise reaches the denoiser with a smaller latent tensor and fails an
    internal shape assertion.  Silence is preferable here: it preserves the
    supplied audio and gives the model a correctly-sized conditioning tensor.
    """
    if not math.isfinite(target_duration) or target_duration <= 0:
        raise ValueError("A2V target duration must be positive")
    if not sources or len(sources) != len(start_times):
        raise ValueError("each A2V audio clip requires one start time")
    if len(sources) > 8:
        raise ValueError("at most 8 A2V audio clips are supported")
    for start in start_times:
        if not math.isfinite(start) or start < 0 or start >= target_duration:
            raise ValueError("A2V audio start time must be within the video")
    filters: list[str] = []
    labels: list[str] = []
    for index, start in enumerate(start_times):
        delay_ms = round(start * 1000)
        label = f"a{index}"
        filters.append(
            f"[{index}:a]aresample=16000,aformat=sample_fmts=s16:channel_layouts=stereo,"
            f"adelay=delays={delay_ms}:all=1[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(
        f"{''.join(labels)}amix=inputs={len(sources)}:duration=longest:dropout_transition=0:normalize=0,"
        f"alimiter=limit=0.95,apad,atrim=duration={target_duration:.9f}[mixed]"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
    ]
    for source in sources:
        command.extend(["-i", str(source)])
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[mixed]", "-vn",
        "-ar", "16000", "-ac", "2", "-c:a", "pcm_s16le", str(destination),
    ])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        raise ValueError(f"failed to normalize A2V audio: {completed.stderr.strip()}")


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
    audios: list[UploadFile] | None = File(None),
    audio_start_times: str = Form("[]"),
    audio: UploadFile | None = File(None),
    audio_max_duration: float = Form(0.0),
    # Backward compatibility for the original single-start-image client.
    image_strength: float = Form(1.0),
    image: UploadFile | None = File(None),
    motion_lora_strength: float = Form(-1.0),
    acceleration: str = Form(""),
) -> FileResponse:
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")
    _validate_dimensions(width, height, num_frames, fps)
    try:
        acceleration_mode = normalize_mode(acceleration or sol_runtime.default_mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
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

    audio_uploads = list(audios or [])
    if audio is not None:
        if audio_uploads:
            temp_dir.cleanup()
            raise HTTPException(400, "use either audios or the legacy audio field, not both")
        audio_uploads = [audio]
        audio_start_times = "[0]"
    audio_path: Path | None = None
    if audio_uploads:
        try:
            parsed_audio_starts = json.loads(audio_start_times)
        except json.JSONDecodeError as exc:
            temp_dir.cleanup()
            raise HTTPException(400, "audio_start_times must be a JSON array") from exc
        if not isinstance(parsed_audio_starts, list) or len(parsed_audio_starts) != len(audio_uploads):
            temp_dir.cleanup()
            raise HTTPException(400, "each audio clip requires one start time")
        try:
            normalized_audio_starts = [float(value) for value in parsed_audio_starts]
        except (TypeError, ValueError) as exc:
            temp_dir.cleanup()
            raise HTTPException(400, "audio start times must be numeric") from exc
        uploaded_audio_paths: list[Path] = []
        total_audio_bytes = 0
        for position, upload in enumerate(audio_uploads):
            suffix = Path(upload.filename or "audio.wav").suffix.lower() or ".wav"
            uploaded_audio_path = root / f"uploaded-audio-{position}{suffix}"
            audio_data = await upload.read()
            total_audio_bytes += len(audio_data)
            if total_audio_bytes > 256 << 20:
                temp_dir.cleanup()
                raise HTTPException(413, "combined audio is too large (max 256 MB)")
            uploaded_audio_path.write_bytes(audio_data)
            uploaded_audio_paths.append(uploaded_audio_path)
        audio_path = root / "input-audio-stereo.wav"
        try:
            await asyncio.to_thread(
                _normalize_a2v_audio,
                uploaded_audio_paths,
                normalized_audio_starts,
                audio_path,
                num_frames / fps,
            )
        except ValueError as exc:
            temp_dir.cleanup()
            raise HTTPException(400, str(exc)) from exc

    output = root / "output.mp4"
    try:
        async with generation_lock:
            if missing_paths(MODEL_DIR):
                raise HTTPException(409, "LTX model files are not ready; open Spark Media settings and prepare the video model")
            if audio_path is not None:
                if a2v_missing_paths(MODEL_DIR):
                    raise HTTPException(409, "LTX A2V model files are not ready; open Spark Media settings and prepare the video model")
                loaded_a2v = await get_a2v_pipeline()
                acceleration_label = await asyncio.to_thread(
                    _generate_a2v, loaded_a2v, prompt, str(output), str(audio_path),
                    num_frames / fps,
                    [str(path) for path in image_paths], normalized_indices, normalized_strengths,
                    width, height, num_frames, fps, seed,
                )
            else:
                loaded = await get_pipeline(effective_lora_path, effective_lora_strength)
                acceleration_label = await asyncio.to_thread(
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
                    acceleration_mode,
                    bool(effective_lora_path),
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
        headers={
            "X-Seed": str(seed),
            "X-LTX-Acceleration": acceleration_label,
        },
    )


class _CleanupTask:
    def __init__(self, temp_dir: tempfile.TemporaryDirectory[str]):
        self.temp_dir = temp_dir

    async def __call__(self) -> None:
        self.temp_dir.cleanup()
