#!/usr/bin/env python3
"""Small, persistent API wrapper around Ostris ai-toolkit's CLI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field


DATA_ROOT = Path(os.getenv("TRAINER_DATA_ROOT", "/data")).resolve()
DATASETS_ROOT = DATA_ROOT / "datasets"
JOBS_ROOT = DATA_ROOT / "jobs"
OUTPUT_ROOT = DATA_ROOT / "output"
TOOLKIT_ROOT = Path(os.getenv("AI_TOOLKIT_ROOT", "/opt/ai-toolkit")).resolve()
REGISTERED_ROOT = Path(os.getenv("REGISTERED_LORA_ROOT", "/registered-loras")).resolve()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
STEP_PATTERNS = (
    re.compile(r"(?:step|steps?)\s*[:=]?\s*(\d+)\s*/\s*(\d+)", re.I),
    re.compile(r"(\d+)\s*/\s*(\d+).*?(?:loss|lr)", re.I),
)
LOSS_PATTERN = re.compile(r"loss\s*[:=]\s*([0-9.eE+-]+)", re.I)
CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY", "").strip()
CIVITAI_TOKEN_FILE = Path("/root/.cache/huggingface/media-secrets/civitai_token")
MAX_LORA_BYTES = 2 * 1024 * 1024 * 1024

for directory in (DATASETS_ROOT, JOBS_ROOT, OUTPUT_ROOT, REGISTERED_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Krea 2 LoRA Trainer API")
process_lock = asyncio.Lock()
active_process: asyncio.subprocess.Process | None = None
active_job_id: str | None = None


class DatasetCreate(BaseModel):
    name: str


class CaptionUpdate(BaseModel):
    caption: str = Field(max_length=8192)


class TrainRequest(BaseModel):
    name: str
    dataset: str
    trigger_word: str = Field(default="", max_length=128)
    steps: int = Field(default=1500, ge=100, le=10000)
    rank: int = Field(default=32, ge=4, le=128)
    alpha: int = Field(default=32, ge=1, le=128)
    learning_rate: float = Field(default=1e-4, gt=0, le=0.01)
    resolutions: list[int] = Field(default_factory=lambda: [512, 768, 1024])
    caption_dropout: float = Field(default=0.05, ge=0, le=0.5)
    save_every: int = Field(default=250, ge=50, le=5000)
    sample_prompt: str = Field(default="", max_length=2048)


class CivitaiImportRequest(BaseModel):
    source: str = Field(min_length=1, max_length=2048)
    name: str = Field(default="", max_length=64)
    trigger_word: str = Field(default="", max_length=256)
    civitai_token: str = Field(default="", max_length=512)


def checked_name(value: str, label: str = "name") -> str:
    value = value.strip()
    if not SAFE_NAME.fullmatch(value):
        raise HTTPException(400, f"{label} must use 1-64 letters, digits, dot, dash, or underscore")
    return value


def dataset_path(name: str) -> Path:
    return DATASETS_ROOT / checked_name(name, "dataset name")


def job_path(job_id: str) -> Path:
    return JOBS_ROOT / f"{checked_name(job_id, 'job id')}.json"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def stored_civitai_token() -> str:
    try:
        return CIVITAI_TOKEN_FILE.read_text().strip()
    except OSError:
        return CIVITAI_API_KEY


def save_civitai_token(token: str) -> None:
    CIVITAI_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    CIVITAI_TOKEN_FILE.write_text(token)
    CIVITAI_TOKEN_FILE.chmod(0o600)


def civitai_request(url: str, token: str) -> urllib.request.Request:
    request = urllib.request.Request(url, headers={"User-Agent": "Media-Krea2-LoRA/1.0"})
    request.add_header("Authorization", f"Bearer {token}")
    return request


def civitai_json(path: str, token: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(civitai_request(f"{CIVITAI_API_BASE}/{path.lstrip('/')}", token), timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise ValueError("Civitai API key was rejected") from exc
        raise ValueError(f"Civitai API returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ValueError(f"Civitai API request failed: {exc}") from exc


def civitai_version_id(source: str, token: str) -> str:
    source = source.strip()
    if source.isdigit():
        return source
    parsed = urllib.parse.urlparse(source)
    if parsed.hostname not in {"civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"}:
        raise ValueError("use a Civitai URL or numeric model-version ID")
    query = urllib.parse.parse_qs(parsed.query)
    version = (query.get("modelVersionId") or query.get("modelversionid") or [""])[0]
    if version.isdigit():
        return version
    match = re.search(r"/(?:api/download/models|model-versions)/(\d+)", parsed.path, re.I)
    if match:
        return match.group(1)
    model = re.search(r"/models/(\d+)", parsed.path, re.I)
    if not model:
        raise ValueError("the Civitai URL does not contain a model or version ID")
    metadata = civitai_json(f"models/{model.group(1)}", token)
    versions = metadata.get("modelVersions") or []
    if not versions:
        raise ValueError("the Civitai model has no downloadable version")
    return str(versions[0]["id"])


def safe_lora_filename(value: str) -> str:
    stem = Path(value).stem
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip(".-")
    if not stem:
        stem = "civitai-lora"
    return f"{stem[:64]}.safetensors"


def import_civitai_lora(request: CivitaiImportRequest) -> dict[str, Any]:
    token = request.civitai_token.strip() or stored_civitai_token()
    if not token:
        raise ValueError("Civitai API key is not configured")
    if len(token) < 16 or any(character.isspace() for character in token):
        raise ValueError("invalid Civitai API key")
    if request.civitai_token.strip():
        save_civitai_token(token)
    version_id = civitai_version_id(request.source, token)
    metadata = civitai_json(f"model-versions/{version_id}", token)
    if str((metadata.get("model") or {}).get("type", "")).lower() != "lora":
        raise ValueError("the selected Civitai model is not a LoRA")
    if "krea 2" not in str(metadata.get("baseModel", "")).lower():
        raise ValueError("the selected LoRA is not based on Krea 2")
    files = [
        item for item in metadata.get("files", [])
        if str(item.get("name", "")).lower().endswith(".safetensors") and item.get("downloadUrl")
    ]
    if not files:
        raise ValueError("this model version has no safetensors file")
    selected = next((item for item in files if item.get("primary")), files[0])
    filename = safe_lora_filename(request.name.strip() or selected.get("name", ""))
    destination = REGISTERED_ROOT / filename
    if destination.exists():
        raise FileExistsError(f"{filename} is already registered")
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(civitai_request(selected["downloadUrl"], token), timeout=600) as response:
            declared = int(response.headers.get("Content-Length") or 0)
            if declared > MAX_LORA_BYTES:
                raise ValueError("LoRA file exceeds the 2 GiB limit")
            total = 0
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_LORA_BYTES:
                        raise ValueError("LoRA file exceeds the 2 GiB limit")
                    output.write(chunk)
                    digest.update(chunk)
        expected_sha = str((selected.get("hashes") or {}).get("SHA256", "")).lower()
        if expected_sha and digest.hexdigest().lower() != expected_sha:
            raise ValueError("Civitai LoRA checksum mismatch")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    trained_words = [str(word).strip() for word in metadata.get("trainedWords", []) if str(word).strip()]
    trigger_word = request.trigger_word.strip() or ", ".join(trained_words)
    saved_metadata = {
        "name": request.name.strip() or Path(filename).stem,
        "trigger_word": trigger_word,
        "trained_words": trained_words,
        "rank": None,
        "created_at": time.time(),
        "source": request.source.strip(),
        "civitai_version_id": version_id,
        "base_model": metadata.get("baseModel", ""),
    }
    write_json(destination.with_suffix(".json"), saved_metadata)
    return {**saved_metadata, "filename": filename, "size": destination.stat().st_size}


def recover_interrupted_jobs() -> None:
    """A container restart kills the CLI child, so do not leave phantom jobs active."""
    for path in JOBS_ROOT.glob("*.json"):
        job = read_json(path, {})
        if job.get("status") in {"queued", "running"}:
            job.update({
                "status": "failed",
                "error": "trainer service restarted while this job was active",
                "finished_at": time.time(),
            })
            write_json(path, job)


recover_interrupted_jobs()


def dataset_info(path: Path) -> dict[str, Any]:
    images = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    items = []
    captioned = 0
    for image in images:
        caption_path = image.with_suffix(".txt")
        caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else ""
        captioned += bool(caption)
        items.append({"name": image.name, "caption": caption, "url": f"/datasets/{path.name}/images/{image.name}"})
    return {"name": path.name, "images": len(images), "captioned": captioned, "items": items}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "toolkit": str(TOOLKIT_ROOT), "active_job": active_job_id}


@app.get("/datasets")
async def list_datasets() -> list[dict[str, Any]]:
    return [dataset_info(path) for path in sorted(DATASETS_ROOT.iterdir()) if path.is_dir()]


@app.post("/datasets", status_code=201)
async def create_dataset(request: DatasetCreate) -> dict[str, Any]:
    path = dataset_path(request.name)
    if path.exists():
        raise HTTPException(409, "dataset already exists")
    path.mkdir(parents=True)
    return dataset_info(path)


@app.delete("/datasets/{name}", status_code=204)
async def delete_dataset(name: str) -> None:
    path = dataset_path(name)
    if not path.is_dir():
        raise HTTPException(404, "dataset not found")
    if active_job_id is not None:
        job = read_json(job_path(active_job_id), {})
        if job.get("dataset") == name and job.get("status") in {"queued", "running"}:
            raise HTTPException(409, "the active training job is using this dataset")
    shutil.rmtree(path)


@app.post("/datasets/{name}/images")
async def upload_images(
    name: str,
    images: list[UploadFile] = File(...),
    default_caption: str = Form(default=""),
) -> dict[str, Any]:
    path = dataset_path(name)
    if not path.is_dir():
        raise HTTPException(404, "dataset not found")
    if len(images) > 200:
        raise HTTPException(400, "at most 200 images may be uploaded at once")
    for upload in images:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            raise HTTPException(400, f"unsupported image type: {upload.filename}")
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", Path(upload.filename or "image").stem).strip(".-") or "image"
        destination = path / f"{stem[:52]}-{uuid.uuid4().hex[:8]}{suffix}"
        total = 0
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > 32 * 1024 * 1024:
                    output.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(413, f"image too large: {upload.filename}")
                output.write(chunk)
        if default_caption.strip():
            destination.with_suffix(".txt").write_text(default_caption.strip(), encoding="utf-8")
    return dataset_info(path)


@app.get("/datasets/{name}/images/{filename}")
async def get_dataset_image(name: str, filename: str):
    from fastapi.responses import FileResponse

    path = dataset_path(name) / Path(filename).name
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(404, "image not found")
    return FileResponse(path)


@app.put("/datasets/{name}/images/{filename}/caption")
async def update_caption(name: str, filename: str, request: CaptionUpdate) -> dict[str, str]:
    image = dataset_path(name) / Path(filename).name
    if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(404, "image not found")
    image.with_suffix(".txt").write_text(request.caption.strip(), encoding="utf-8")
    return {"status": "saved"}


@app.delete("/datasets/{name}/images/{filename}", status_code=204)
async def delete_dataset_image(name: str, filename: str) -> None:
    image = dataset_path(name) / Path(filename).name
    if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(404, "image not found")
    image.unlink()
    image.with_suffix(".txt").unlink(missing_ok=True)


def build_config(job_id: str, request: TrainRequest, dataset: Path) -> dict[str, Any]:
    sample_prompt = request.sample_prompt.strip() or f"a detailed portrait of {request.trigger_word or 'the trained subject'}"
    return {
        "job": "extension",
        "config": {
            "name": request.name,
            "process": [{
                "type": "diffusion_trainer",
                "training_folder": str(OUTPUT_ROOT / job_id),
                "device": "cuda:0",
                "trigger_word": request.trigger_word.strip() or None,
                "network": {"type": "lora", "linear": request.rank, "linear_alpha": request.alpha},
                "save": {
                    "dtype": "bf16", "save_every": request.save_every,
                    "max_step_saves_to_keep": 4, "save_format": "safetensors", "push_to_hub": False,
                },
                "datasets": [{
                    "folder_path": str(dataset), "caption_ext": "txt",
                    "caption_dropout_rate": request.caption_dropout, "shuffle_tokens": False,
                    "cache_latents_to_disk": True, "resolution": request.resolutions,
                }],
                "train": {
                    "batch_size": 1, "steps": request.steps, "gradient_accumulation": 1,
                    "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch", "optimizer": "adamw8bit",
                    "timestep_type": "linear", "lr": request.learning_rate,
                    "dtype": "bf16", "cache_text_embeddings": True,
                    "unload_text_encoder": True, "disable_sampling": False,
                },
                "model": {
                    "name_or_path": "krea/Krea-2-Turbo", "arch": "krea2:turbo",
                    "quantize": True, "qtype": "qfloat8", "quantize_te": True,
                    "qtype_te": "qfloat8", "low_vram": False, "layer_offloading": False,
                    "assistant_lora_path": "ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors",
                    "model_kwargs": {}, "compile": False,
                },
                "sample": {
                    "sampler": "flowmatch", "sample_every": request.save_every,
                    "sample_start_step": 0, "width": 1024, "height": 1024,
                    "samples": [{"prompt": sample_prompt}], "neg": "", "seed": 42,
                    "walk_seed": False, "guidance_scale": 1, "sample_steps": 9,
                },
                "logging": {"log_every": 1, "use_ui_logger": False},
            }],
        },
        "meta": {"name": "[name]", "version": "1.0", "trainer": "media-krea2-lora"},
    }


def update_progress(job: dict[str, Any], line: str) -> None:
    for pattern in STEP_PATTERNS:
        match = pattern.search(line)
        if match:
            job["step"], job["total_steps"] = int(match.group(1)), int(match.group(2))
            break
    loss = LOSS_PATTERN.search(line)
    if loss:
        try:
            job["loss"] = float(loss.group(1))
        except ValueError:
            pass


def find_result(output_dir: Path, name: str) -> Path | None:
    candidates = [p for p in output_dir.rglob("*.safetensors") if "step" not in p.stem.lower()]
    named = [p for p in candidates if name.lower() in p.stem.lower()]
    pool = named or candidates
    return max(pool, key=lambda path: path.stat().st_mtime) if pool else None


async def run_training(job_id: str, request: TrainRequest, config_path: Path) -> None:
    global active_process, active_job_id
    state_path = job_path(job_id)
    job = read_json(state_path, {})
    log_path = JOBS_ROOT / f"{job_id}.log"
    try:
        async with process_lock:
            active_job_id = job_id
            job.update({"status": "running", "started_at": time.time()})
            write_json(state_path, job)
            active_process = await asyncio.create_subprocess_exec(
                "python", str(TOOLKIT_ROOT / "run.py"), str(config_path),
                cwd=str(TOOLKIT_ROOT), stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, start_new_session=True,
            )
            assert active_process.stdout is not None
            with log_path.open("a", encoding="utf-8") as log_file:
                async for raw in active_process.stdout:
                    line = raw.decode("utf-8", errors="replace")
                    log_file.write(line)
                    log_file.flush()
                    update_progress(job, line)
                    job["last_message"] = line.strip()[-500:]
                    write_json(state_path, job)
            return_code = await active_process.wait()
            # The cancellation endpoint updates the persisted copy while this
            # task owns its in-memory progress object.
            job = read_json(state_path, job)
            if job.get("status") == "cancelled":
                return
            if return_code != 0:
                job.update({"status": "failed", "error": f"ai-toolkit exited with code {return_code}"})
                write_json(state_path, job)
                return
            result = find_result(OUTPUT_ROOT / job_id, request.name)
            if result is None:
                job.update({"status": "failed", "error": "training completed but no LoRA file was found"})
                write_json(state_path, job)
                return
            registered = REGISTERED_ROOT / f"{request.name}.safetensors"
            shutil.copy2(result, registered)
            metadata = {
                "name": request.name, "trigger_word": request.trigger_word.strip(),
                "rank": request.rank, "alpha": request.alpha, "dataset": request.dataset,
                "created_at": time.time(), "source": str(result),
            }
            registered.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            job.update({"status": "completed", "finished_at": time.time(), "lora": registered.name})
            write_json(state_path, job)
    except Exception as exc:
        job.update({"status": "failed", "error": str(exc), "finished_at": time.time()})
        write_json(state_path, job)
    finally:
        active_process = None
        active_job_id = None


@app.get("/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    jobs = [read_json(path, {}) for path in JOBS_ROOT.glob("*.json")]
    return sorted((job for job in jobs if job), key=lambda job: job.get("created_at", 0), reverse=True)


@app.post("/jobs", status_code=202)
async def create_job(request: TrainRequest) -> dict[str, Any]:
    global active_job_id
    if active_job_id is not None:
        raise HTTPException(409, "another training job is already running")
    request.name = checked_name(request.name, "LoRA name")
    dataset = dataset_path(request.dataset)
    if not dataset.is_dir():
        raise HTTPException(404, "dataset not found")
    info = dataset_info(dataset)
    if info["images"] < 2:
        raise HTTPException(400, "at least two training images are required")
    if info["captioned"] != info["images"]:
        raise HTTPException(400, "every training image needs a caption")
    if any(value not in {256, 384, 512, 768, 1024, 1280} for value in request.resolutions):
        raise HTTPException(400, "unsupported training resolution")
    if len(set(request.resolutions)) != len(request.resolutions) or not request.resolutions:
        raise HTTPException(400, "training resolutions must be unique and non-empty")
    if (REGISTERED_ROOT / f"{request.name}.safetensors").exists():
        raise HTTPException(409, "a registered LoRA with that name already exists")
    job_id = uuid.uuid4().hex
    config = build_config(job_id, request, dataset)
    config_path = JOBS_ROOT / f"{job_id}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    job = {
        "id": job_id, "name": request.name, "dataset": request.dataset,
        "status": "queued", "step": 0, "total_steps": request.steps,
        "created_at": time.time(), "config": config_path.name,
    }
    write_json(job_path(job_id), job)
    active_job_id = job_id
    asyncio.create_task(run_training(job_id, request, config_path))
    return job


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    global active_process
    path = job_path(job_id)
    job = read_json(path, None)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.get("status") not in {"queued", "running"}:
        raise HTTPException(409, "job is not active")
    job.update({"status": "cancelled", "finished_at": time.time()})
    write_json(path, job)
    if active_job_id == job_id and active_process is not None and active_process.returncode is None:
        try:
            os.killpg(active_process.pid, signal.SIGTERM)
            await asyncio.wait_for(active_process.wait(), timeout=10)
        except asyncio.TimeoutError:
            os.killpg(active_process.pid, signal.SIGKILL)
    return job


@app.post("/loras/import", status_code=201)
async def import_lora(request: CivitaiImportRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(import_civitai_lora, request)
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/loras")
async def list_loras() -> list[dict[str, Any]]:
    result = []
    for path in sorted(REGISTERED_ROOT.glob("*.safetensors")):
        if path.name == "skc3vo.safetensors":
            continue
        metadata = read_json(path.with_suffix(".json"), {})
        result.append({**metadata, "filename": path.name, "size": path.stat().st_size})
    return result


@app.delete("/loras/{filename}", status_code=204)
async def delete_lora(filename: str) -> None:
    filename = Path(filename).name
    if not filename.endswith(".safetensors"):
        raise HTTPException(400, "LoRA filename must end in .safetensors")
    path = REGISTERED_ROOT / filename
    if not path.is_file():
        raise HTTPException(404, "LoRA not found")
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
