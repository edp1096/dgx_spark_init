"""Generation functions — thin shims that submit tasks to the worker process."""

import copy
import json
import logging
import tempfile
import time
from pathlib import Path

import gradio as gr

from pipeline_manager import OUTPUT_DIR, REQUIRED_MODELS
from worker import WorkerProcessManager

logger = logging.getLogger("zifk-ui")

# ---------------------------------------------------------------------------
# Worker process singleton
# ---------------------------------------------------------------------------
_worker_mgr: WorkerProcessManager | None = None


def get_worker_mgr() -> WorkerProcessManager:
    global _worker_mgr
    if _worker_mgr is None:
        from zifk_config import MODEL_DIR
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


_GEN_TAB_TYPES = {"generate": {"zit_t2i", "zib_t2i", "klein_t2i", "klein_base_t2i"},
                   "edit": {"klein_edit", "klein_multiref"}}


def get_gen_info_for_tab(gen_type: str) -> str:
    if _gen_active:
        return "Generating..."
    result = _last_gen_result
    if result is None:
        return ""
    allowed = _GEN_TAB_TYPES.get(gen_type)
    if allowed:
        if result["gen_type"] not in allowed:
            return ""
    elif result["gen_type"] != gen_type:
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
                    paths = payload["paths"]
                    seed = payload["seed"]
                    if len(paths) == 1:
                        info = f"Seed: {seed} | Time: {elapsed:.1f}s | Output: {Path(paths[0]).name}"
                    else:
                        info = f"Seed: {seed} | Time: {elapsed:.1f}s | Images: {len(paths)}"
                    logger.info("Generation complete: %s", info)
                    _save_metadata(paths[0], gen_type, seed, elapsed, kwargs)
                    _gen_active = False
                    _result_version += 1
                    _last_gen_result = {
                        "paths": paths, "info": info,
                        "gen_type": gen_type, "time": time.time(),
                    }
                    return paths, info
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
# Generation: Z-Image T2I
# ---------------------------------------------------------------------------
def generate_zimage_t2i(
    prompt, model_type, resolution, seed, num_images,
    negative_prompt="", num_steps=None, guidance_scale=None,
    cfg_normalization=False, cfg_truncation=1.0,
    max_sequence_length=512, attention_backend=None,
    progress=gr.Progress(track_tqdm=True),
):
    gen_type = "zit_t2i" if model_type == "ZIT (Fast)" else "zib_t2i"
    _validate(gen_type, prompt)

    is_turbo = model_type == "ZIT (Fast)"
    if num_steps is None:
        num_steps = 8 if is_turbo else 28
    if guidance_scale is None:
        guidance_scale = 0.0 if is_turbo else 3.5

    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "model_type": "turbo" if is_turbo else "base",
        "negative_prompt": negative_prompt if not is_turbo else None,
        "width": int(w), "height": int(h),
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "cfg_normalization": bool(cfg_normalization),
        "cfg_truncation": float(cfg_truncation),
        "num_images": int(num_images),
        "max_sequence_length": int(max_sequence_length),
        "seed": int(seed),
    }
    if attention_backend:
        kwargs["attention_backend"] = attention_backend
    return _submit_and_wait(gen_type, kwargs, progress)


# ---------------------------------------------------------------------------
# Generation: Klein T2I (Distilled)
# ---------------------------------------------------------------------------
def generate_klein_t2i(
    prompt, resolution, seed, num_steps=4, guidance=1.0,
    progress=gr.Progress(track_tqdm=True),
):
    _validate("klein_t2i", prompt)
    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "width": int(w), "height": int(h),
        "num_steps": int(num_steps),
        "guidance": float(guidance),
        "seed": int(seed),
    }
    return _submit_and_wait("klein_t2i", kwargs, progress)


# ---------------------------------------------------------------------------
# Generation: Klein Base T2I (Non-distilled, CFG)
# ---------------------------------------------------------------------------
def generate_klein_base_t2i(
    prompt, resolution, seed, num_steps=50, guidance=4.0,
    progress=gr.Progress(track_tqdm=True),
):
    _validate("klein_base_t2i", prompt)
    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "width": int(w), "height": int(h),
        "num_steps": int(num_steps),
        "guidance": float(guidance),
        "seed": int(seed),
    }
    return _submit_and_wait("klein_base_t2i", kwargs, progress)


# ---------------------------------------------------------------------------
# Generation: Klein Edit (single reference)
# ---------------------------------------------------------------------------
def generate_klein_edit(
    prompt, image, resolution, seed, klein_variant="flux.2-klein-4b",
    num_steps=4, guidance=1.0,
    progress=gr.Progress(track_tqdm=True),
):
    _validate("klein_edit", prompt)
    if image is None:
        raise gr.Error("Input image is required.")

    from PIL import Image as PILImage
    import numpy as np
    if isinstance(image, np.ndarray):
        img = PILImage.fromarray(image)
    else:
        img = image
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
    img.save(tmp.name)
    tmp.close()

    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "image_path": tmp.name,
        "width": int(w), "height": int(h),
        "num_steps": int(num_steps),
        "guidance": float(guidance),
        "klein_variant": klein_variant,
        "seed": int(seed),
    }
    return _submit_and_wait("klein_edit", kwargs, progress)


# ---------------------------------------------------------------------------
# Generation: Klein Multi-Reference
# ---------------------------------------------------------------------------
def generate_klein_multiref(
    prompt, images, resolution, seed, klein_variant="flux.2-klein-4b",
    num_steps=4, guidance=1.0,
    progress=gr.Progress(track_tqdm=True),
):
    _validate("klein_multiref", prompt)
    if not images or len(images) == 0:
        raise gr.Error("At least one reference image is required.")

    from PIL import Image as PILImage
    import numpy as np

    image_paths = []
    for img in images:
        if isinstance(img, np.ndarray):
            pil_img = PILImage.fromarray(img)
        else:
            pil_img = img
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
        pil_img.save(tmp.name)
        tmp.close()
        image_paths.append(tmp.name)

    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "image_paths": image_paths,
        "width": int(w), "height": int(h),
        "num_steps": int(num_steps),
        "guidance": float(guidance),
        "klein_variant": klein_variant,
        "seed": int(seed),
    }
    return _submit_and_wait("klein_multiref", kwargs, progress)


# ---------------------------------------------------------------------------
# Generation: Compare (multi-model)
# ---------------------------------------------------------------------------
def generate_compare(
    prompt, resolution, seed, use_zit, use_zib, use_klein, use_klein_base,
    negative_prompt="", zit_steps=8, zit_guidance=0.0,
    zib_steps=28, zib_cfg=3.5,
    klein_steps=4, klein_guidance=1.0,
    klein_base_steps=50, klein_base_guidance=4.0,
    progress=gr.Progress(track_tqdm=True),
):
    if not any([use_zit, use_zib, use_klein, use_klein_base]):
        raise gr.Error("Select at least one model for comparison.")
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is required.")

    results = []

    if use_zit:
        try:
            paths, info = generate_zimage_t2i(
                prompt, "ZIT (Fast)", resolution, seed, 1,
                num_steps=zit_steps, guidance_scale=zit_guidance,
                progress=progress,
            )
            results.append((paths[0], f"ZIT | {info}"))
        except Exception as e:
            results.append((None, f"ZIT | Error: {e}"))

    if use_zib:
        try:
            paths, info = generate_zimage_t2i(
                prompt, "ZIB (Creative)", resolution, seed, 1,
                negative_prompt=negative_prompt,
                num_steps=zib_steps, guidance_scale=zib_cfg,
                progress=progress,
            )
            results.append((paths[0], f"ZIB | {info}"))
        except Exception as e:
            results.append((None, f"ZIB | Error: {e}"))

    if use_klein:
        try:
            paths, info = generate_klein_t2i(
                prompt, resolution, seed,
                num_steps=klein_steps, guidance=klein_guidance,
                progress=progress,
            )
            results.append((paths[0], f"Klein | {info}"))
        except Exception as e:
            results.append((None, f"Klein | Error: {e}"))

    if use_klein_base:
        try:
            paths, info = generate_klein_base_t2i(
                prompt, resolution, seed,
                num_steps=klein_base_steps, guidance=klein_base_guidance,
                progress=progress,
            )
            results.append((paths[0], f"Klein Base | {info}"))
        except Exception as e:
            results.append((None, f"Klein Base | Error: {e}"))

    return results


# ---------------------------------------------------------------------------
# Helper: match resolution from image
# ---------------------------------------------------------------------------
def match_image_resolution(image) -> str:
    """Get resolution string from image, snapped to nearest multiple of 16."""
    if image is None:
        return "1024x1024"
    import numpy as np
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size
    w = (w // 16) * 16
    h = (h // 16) * 16
    w = max(w, 64)
    h = max(h, 64)
    return f"{w}x{h}"
