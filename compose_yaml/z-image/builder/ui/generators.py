"""Image generation functions — thin shims that submit tasks to the worker process."""

import copy
import hashlib
import json
import logging
import tempfile
import time
from pathlib import Path

import gradio as gr

from pipeline_manager import OUTPUT_DIR, REQUIRED_MODELS
from worker import WorkerProcessManager

logger = logging.getLogger("zimage-ui")

# ---------------------------------------------------------------------------
# Worker process singleton
# ---------------------------------------------------------------------------
_worker_mgr: WorkerProcessManager | None = None


def get_worker_mgr() -> WorkerProcessManager:
    global _worker_mgr
    if _worker_mgr is None:
        from config import MODEL_DIR
        _worker_mgr = WorkerProcessManager(str(MODEL_DIR))
    return _worker_mgr


def set_model_dir(model_dir: str):
    mgr = get_worker_mgr()
    mgr.model_dir = model_dir
    if mgr.is_alive():
        mgr.kill()


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_loading_status = ""
_gen_active = False
_last_gen_result: dict | None = None
_result_version = 0


def get_loading_status() -> str:
    return _loading_status


def is_generation_active() -> bool:
    return _gen_active


def get_result_version() -> int:
    return _result_version


def get_gen_info_for_tab(gen_type: str) -> str:
    if _gen_active:
        return "Generating..."
    result = _last_gen_result
    if result is None or result["gen_type"] != gen_type:
        return ""
    if time.time() - result["time"] > 600:
        return ""
    return result["info"]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def _sanitize_kwargs(kwargs: dict) -> dict:
    data = copy.deepcopy(kwargs)
    for key, val in list(data.items()):
        if isinstance(val, (bytes, bytearray)):
            data[key] = "<binary>"
        elif hasattr(val, "tolist"):
            data[key] = "<array>"
    return data


def _save_metadata(path: str, gen_type: str, seed: int, elapsed: float, kwargs: dict):
    try:
        metadata = {
            "gen_type": gen_type,
            "seed": seed,
            "elapsed": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kwargs": _sanitize_kwargs(kwargs),
        }
        json_path = Path(path).with_suffix(".json")
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.warning("Failed to save metadata: %s", e)


# ---------------------------------------------------------------------------
# Submit and wait
# ---------------------------------------------------------------------------
def _submit_and_wait(gen_type: str, kwargs: dict, progress) -> tuple[str, str]:
    global _loading_status, _gen_active, _last_gen_result, _result_version

    _gen_active = True
    mgr = get_worker_mgr()
    mgr.ensure_running()
    task_id = mgr.submit_task(gen_type, kwargs)

    _loading_status = "**Starting generation...**"
    start = time.time()

    try:
        while True:
            if not mgr.is_alive():
                _loading_status = ""
                raise gr.Error("Worker process crashed. Click Generate to restart.")

            for msg in mgr.poll_progress():
                if msg.get("task_id") != task_id:
                    continue
                mtype = msg.get("type")
                data = msg.get("data", {})
                if mtype == "loading_start":
                    name = data.get("name", "?")
                    _loading_status = f"**Loading {name}...**"
                    if progress:
                        progress(0.1, desc=f"Loading {name}")
                elif mtype == "loading_done":
                    name = data.get("name", "?")
                    _loading_status = f"**{name} loaded**"
                    if progress:
                        progress(0.5, desc=f"{name} loaded")

            result = mgr.get_result(timeout=0.2)
            if result and result.get("task_id") == task_id:
                elapsed = time.time() - start
                _loading_status = ""

                if result["status"] == "ok":
                    payload = result["payload"]
                    path = payload["path"]
                    seed = payload["seed"]
                    info = f"Seed: {seed} | Time: {elapsed:.1f}s | Output: {Path(path).name}"
                    logger.info("Generation complete: %s", info)
                    _save_metadata(path, gen_type, seed, elapsed, kwargs)
                    _gen_active = False
                    _result_version += 1
                    _last_gen_result = {
                        "path": path, "info": info,
                        "gen_type": gen_type, "time": time.time(),
                    }
                    return path, info
                else:
                    raise gr.Error(f"Generation failed: {result['payload']}")
    except Exception:
        _gen_active = False
        _loading_status = ""
        raise


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(pipeline_type: str, prompt: str):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is required.")
    mgr = get_worker_mgr()
    missing = mgr.check_models(pipeline_type)
    if missing:
        raise gr.Error(f"Missing models: {', '.join(missing)}\nCheck Settings tab.")


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------
def generate_turbo(
    prompt, resolution, seed, num_images, loras,
    progress=gr.Progress(track_tqdm=True),
):
    _validate("turbo", prompt)
    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "width": int(w), "height": int(h),
        "seed": int(seed),
        "num_images": int(num_images),
        "num_steps": 9,
        "loras": loras or [],
    }
    return _submit_and_wait("turbo", kwargs, progress)


def generate_base(
    prompt, negative_prompt, resolution, num_steps, guidance_scale,
    cfg_normalization, cfg_truncation, seed, num_images, loras,
    progress=gr.Progress(track_tqdm=True),
):
    _validate("base", prompt)
    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": int(w), "height": int(h),
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "cfg_normalization": bool(cfg_normalization),
        "cfg_truncation": float(cfg_truncation),
        "seed": int(seed),
        "num_images": int(num_images),
        "loras": loras or [],
    }
    return _submit_and_wait("base", kwargs, progress)


def generate_img2img(
    prompt, negative_prompt, image, strength,
    resolution, num_steps, guidance_scale, seed,
    use_base, loras,
    progress=gr.Progress(track_tqdm=True),
):
    pipeline_type = "img2img_base" if use_base else "img2img_turbo"
    _validate(pipeline_type, prompt)
    if image is None:
        raise gr.Error("Input image is required.")

    # Save image to temp file
    from PIL import Image as PILImage
    import numpy as np
    if isinstance(image, np.ndarray):
        img = PILImage.fromarray(image)
    else:
        img = image
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=OUTPUT_DIR)
    img.save(tmp.name)
    tmp.close()

    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image_path": tmp.name,
        "strength": float(strength),
        "width": int(w), "height": int(h),
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),
        "use_base": bool(use_base),
        "loras": loras or [],
    }
    return _submit_and_wait("img2img", kwargs, progress)
