"""Generation functions for ZIT — thin shims that submit tasks to the worker process."""

import copy
import json
import logging
import time
from pathlib import Path

import gradio as gr

from pipeline_manager import OUTPUT_DIR
from worker import WorkerProcessManager

logger = logging.getLogger("zit-ui")

# ---------------------------------------------------------------------------
# Worker process singleton
# ---------------------------------------------------------------------------
_worker_mgr: WorkerProcessManager | None = None


def get_worker_mgr() -> WorkerProcessManager:
    global _worker_mgr
    if _worker_mgr is None:
        from zit_config import MODEL_DIR
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


_GEN_TAB_TYPES = {
    "generate": {"zit_t2i"},
    "controlnet": {"controlnet"},
    "inpaint": {"inpaint", "outpaint"},
    "faceswap": {"faceswap"},
}


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
# Temp file cleanup
# ---------------------------------------------------------------------------
def _cleanup_temp_files(kwargs: dict):
    """Remove temp files created for IPC (image_path, mask_path, etc.)."""
    for key in ("image_path", "mask_path", "target_path", "source_path"):
        path = kwargs.get(key)
        if path and Path(path).name.startswith("tmp"):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass


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
                    _cleanup_temp_files(kwargs)
                    return paths, info
                else:
                    _cleanup_temp_files(kwargs)
                    raise gr.Error(f"Generation failed: {result['payload']}")
    except Exception:
        _gen_active = False
        _loading_status = ""
        _cleanup_temp_files(kwargs)
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
# Generation: ZIT T2I
# ---------------------------------------------------------------------------
def generate_zit_t2i(
    prompt, resolution, seed, num_images,
    negative_prompt="", num_steps=8, guidance_scale=0.0,
    cfg_normalization=False, cfg_truncation=1.0,
    max_sequence_length=512, attention_backend=None,
    time_shift=3.0,
    lora_name=None, lora_scale=1.0,
    progress=gr.Progress(track_tqdm=True),
):
    gen_type = "zit_t2i"
    _validate(gen_type, prompt)

    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "model_type": "turbo",
        "negative_prompt": negative_prompt or None,
        "width": int(w), "height": int(h),
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "cfg_normalization": bool(cfg_normalization),
        "cfg_truncation": float(cfg_truncation),
        "num_images": int(num_images),
        "max_sequence_length": int(max_sequence_length),
        "time_shift": float(time_shift),
        "seed": int(seed),
        "lora_name": lora_name or None,
        "lora_scale": float(lora_scale),
    }
    if attention_backend:
        kwargs["attention_backend"] = attention_backend
    return _submit_and_wait(gen_type, kwargs, progress)


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


# ---------------------------------------------------------------------------
# Generation: ControlNet
# ---------------------------------------------------------------------------
def generate_controlnet(
    prompt, control_mode, control_image, resolution, seed,
    negative_prompt="", num_steps=8, guidance_scale=0.5,
    cfg_truncation=0.9, control_scale=0.65,
    max_sequence_length=512, time_shift=3.0,
    lora_name=None, lora_scale=1.0,
    progress=gr.Progress(track_tqdm=True),
):
    _validate("controlnet", prompt)
    if control_image is None:
        raise gr.Error("Control image is required.")

    import tempfile
    from PIL import Image as PILImage
    import numpy as np

    # Save preprocessed control image to temp file for IPC
    if isinstance(control_image, np.ndarray):
        img = PILImage.fromarray(control_image)
    else:
        img = control_image
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
    img.save(tmp.name)
    tmp.close()

    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "control_image_path": tmp.name,
        "control_mode": control_mode,
        "control_scale": float(control_scale),
        "width": int(w), "height": int(h),
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "cfg_truncation": float(cfg_truncation),
        "max_sequence_length": int(max_sequence_length),
        "time_shift": float(time_shift),
        "seed": int(seed),
        "lora_name": lora_name or None,
        "lora_scale": float(lora_scale),
    }
    return _submit_and_wait("controlnet", kwargs, progress)


# ---------------------------------------------------------------------------
# Generation: Inpaint
# ---------------------------------------------------------------------------
def generate_inpaint(
    prompt, editor_value, resolution, seed,
    negative_prompt="", num_steps=25, guidance_scale=4.0,
    cfg_truncation=1.0, control_scale=0.9,
    max_sequence_length=512, time_shift=3.0,
    progress=gr.Progress(track_tqdm=True),
):
    _validate("inpaint", prompt)
    if editor_value is None:
        raise gr.Error("Image with mask is required.")

    import tempfile
    from PIL import Image as PILImage
    import numpy as np

    # Extract image and mask from gr.ImageEditor output
    # Format: {"background": ndarray, "layers": [ndarray], "composite": ndarray}
    if isinstance(editor_value, dict):
        background = editor_value.get("background")
        layers = editor_value.get("layers", [])
        if background is None:
            raise gr.Error("No image loaded in editor.")
        # Mask: any non-zero pixel in layers = white (regenerate)
        if layers and len(layers) > 0:
            mask = np.zeros(background.shape[:2], dtype=np.uint8)
            for layer in layers:
                if isinstance(layer, np.ndarray) and layer.ndim >= 2:
                    if layer.ndim == 3:
                        layer_gray = np.any(layer > 0, axis=2).astype(np.uint8) * 255
                    else:
                        layer_gray = (layer > 0).astype(np.uint8) * 255
                    mask = np.maximum(mask, layer_gray)
        else:
            raise gr.Error("Draw a mask on the image first.")
        image = PILImage.fromarray(background)
        mask_img = PILImage.fromarray(mask)
    else:
        raise gr.Error("Unexpected editor format.")

    # Save to temp files for IPC
    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
    image.save(tmp_img.name)
    tmp_img.close()

    tmp_mask = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
    mask_img.save(tmp_mask.name)
    tmp_mask.close()

    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "image_path": tmp_img.name,
        "mask_path": tmp_mask.name,
        "control_scale": float(control_scale),
        "width": int(w), "height": int(h),
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "cfg_truncation": float(cfg_truncation),
        "max_sequence_length": int(max_sequence_length),
        "time_shift": float(time_shift),
        "seed": int(seed),
    }
    return _submit_and_wait("inpaint", kwargs, progress)


# ---------------------------------------------------------------------------
# Generation: Outpaint
# ---------------------------------------------------------------------------
def generate_outpaint(
    prompt, image, direction, expand_px, resolution, seed,
    negative_prompt="", num_steps=25, guidance_scale=4.0,
    cfg_truncation=1.0, control_scale=0.9,
    max_sequence_length=512, time_shift=3.0,
    progress=gr.Progress(track_tqdm=True),
):
    logger.info("generate_outpaint called: direction=%s expand_px=%s image_type=%s",
                direction, expand_px, type(image).__name__)
    _validate("outpaint", prompt)
    if image is None:
        raise gr.Error("Image is required.")

    import tempfile
    from PIL import Image as PILImage
    import numpy as np

    if isinstance(image, np.ndarray):
        img = PILImage.fromarray(image)
    else:
        img = image

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
    img.save(tmp.name)
    tmp.close()

    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "image_path": tmp.name,
        "direction": direction,
        "expand_px": int(expand_px),
        "control_scale": float(control_scale),
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "cfg_truncation": float(cfg_truncation),
        "max_sequence_length": int(max_sequence_length),
        "time_shift": float(time_shift),
        "seed": int(seed),
    }
    return _submit_and_wait("outpaint", kwargs, progress)


# ---------------------------------------------------------------------------
# Generation: FaceSwap
# ---------------------------------------------------------------------------
def generate_faceswap(
    image, prompt, face_index=0, padding=1.3, det_threshold=0.5,
    resolution="768x1024", seed=-1,
    negative_prompt="blurry, low quality, artifacts, unnatural skin",
    num_steps=25, guidance_scale=4.0,
    cfg_truncation=1.0, control_scale=0.9,
    max_sequence_length=512, time_shift=3.0,
    progress=gr.Progress(track_tqdm=True),
):
    """Auto-mask face via SCRFD + ZIT Inpaint for face regeneration."""
    if image is None:
        raise gr.Error("Image is required.")
    _validate("faceswap", prompt)

    import tempfile
    from PIL import Image as PILImage
    import numpy as np

    if isinstance(image, np.ndarray):
        img = PILImage.fromarray(image)
    else:
        img = image

    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
    img.save(tmp_img.name)
    tmp_img.close()

    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "image_path": tmp_img.name,
        "face_index": int(face_index),
        "padding": float(padding),
        "det_threshold": float(det_threshold),
        "control_scale": float(control_scale),
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "cfg_truncation": float(cfg_truncation),
        "max_sequence_length": int(max_sequence_length),
        "time_shift": float(time_shift),
        "seed": int(seed),
    }
    return _submit_and_wait("faceswap", kwargs, progress)


# ---------------------------------------------------------------------------
# Preview: Preprocessor (runs in main process)
# ---------------------------------------------------------------------------
def preview_preprocessor(mode: str, image, model_dir: str = None):
    """Run preprocessor and return control map image for preview."""
    if image is None:
        raise gr.Error("Upload an image first.")
    import numpy as np
    if model_dir is None:
        from zit_config import MODEL_DIR
        model_dir = str(MODEL_DIR)
    try:
        from preprocessors import preprocess
        if isinstance(image, np.ndarray):
            result = preprocess(mode, image, model_dir)
        else:
            result = preprocess(mode, np.array(image), model_dir)
        return result
    except ImportError:
        raise gr.Error(f"Preprocessor '{mode}' not available. Check preprocessors/ directory.")
