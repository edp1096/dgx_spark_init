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
_handler_active = False   # True while _submit_and_wait is running
_active_gen_type = ""     # gen_type of current/last generation
_active_gen_ui_params: dict | None = None  # UI params saved at generation start


def _poll_gen_recovery():
    """Drain worker queues when handler died (browser refresh) but gen is active."""
    global _loading_status, _gen_active, _last_gen_result, _result_version, _active_gen_ui_params

    mgr = _worker_mgr
    if mgr is None:
        _gen_active = False
        _loading_status = ""
        return

    if not mgr.is_alive():
        _gen_active = False
        _loading_status = "**Worker crashed — click Generate to restart**"
        return

    for msg in mgr.poll_progress():
        mtype = msg.get("type")
        data = msg.get("data", {})
        if mtype == "loading_start":
            _loading_status = f"**Loading {data.get('name', '?')}...**"
        elif mtype == "loading_done":
            _loading_status = f"**{data.get('name', '?')} loaded**"

    result = mgr.get_result(timeout=0.05)
    if result is None:
        return
    if result.get("status") == "ok":
        payload = result["payload"]
        paths = payload["paths"]
        seed = payload["seed"]
        if len(paths) == 1:
            info = f"Seed: {seed} | Output: {Path(paths[0]).name}"
        else:
            info = f"Seed: {seed} | Images: {len(paths)}"
        _loading_status = ""
        _gen_active = False
        _active_gen_ui_params = None
        _result_version += 1
        _last_gen_result = {
            "paths": paths, "info": info,
            "gen_type": _active_gen_type, "time": time.time(),
        }
    else:
        _loading_status = ""
        _gen_active = False
        _active_gen_ui_params = None


def get_loading_status() -> str:
    if _gen_active and not _handler_active:
        _poll_gen_recovery()
    return _loading_status


def save_gen_ui_params(params: dict):
    """Save UI parameter values at generation start for refresh recovery."""
    global _active_gen_ui_params
    _active_gen_ui_params = params


def get_gen_ui_params() -> tuple[bool, str, dict | None]:
    """Return (is_active, gen_type, ui_params) for page-load recovery."""
    if _gen_active:
        return True, _active_gen_type, _active_gen_ui_params
    return False, "", None


def wait_for_gen_completion(timeout: int = 600):
    """Block until generation completes (for page-load gallery recovery).
    Returns _last_gen_result or None.
    """
    if not _gen_active:
        return None
    for _ in range(timeout):
        if not _gen_active:
            break
        time.sleep(1)
    return _last_gen_result


def get_latest_gallery(gen_type: str):
    """Return latest result image paths for app.load() recovery."""
    result = _last_gen_result
    if result is None:
        return None
    allowed = _GEN_TAB_TYPES.get(gen_type)
    if allowed and result["gen_type"] not in allowed:
        return None
    if time.time() - result["time"] > 600:
        return None
    return result["paths"]


_GEN_TAB_TYPES = {
    "generate": {"zit_t2i", "controlnet"},
    "inpaint": {"inpaint", "outpaint"},
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
    for key in ("image_path", "mask_path", "target_path", "source_path", "original_image_path"):
        path = kwargs.get(key)
        if path and Path(path).name.startswith("tmp"):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Submit and wait
# ---------------------------------------------------------------------------
def _submit_and_wait(gen_type: str, kwargs: dict, progress,
                     need_controlnet: bool = True) -> tuple[str, str]:
    global _loading_status, _gen_active, _last_gen_result, _result_version
    global _handler_active, _active_gen_type, _active_gen_ui_params

    _gen_active = True
    _handler_active = True
    _active_gen_type = gen_type
    mgr = get_worker_mgr()
    mgr.ensure_mode(need_controlnet)
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
                    _active_gen_ui_params = None
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
        _handler_active = False
        _active_gen_ui_params = None
        _loading_status = ""
        _cleanup_temp_files(kwargs)
        raise
    finally:
        _handler_active = False


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
    lora_stack=None,
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
    }
    # Multi-LoRA stack (preferred) or single LoRA (backward compat)
    if lora_stack:
        kwargs["lora_stack"] = lora_stack
    elif lora_name:
        kwargs["lora_stack"] = [{"name": lora_name, "scale": float(lora_scale)}]
    if attention_backend:
        kwargs["attention_backend"] = attention_backend
    return _submit_and_wait(gen_type, kwargs, progress, need_controlnet=False)


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
    negative_prompt="", num_steps=8, guidance_scale=1.0,
    cfg_normalization=False, cfg_truncation=1.0, control_scale=0.7,
    max_sequence_length=512, time_shift=3.0,
    num_images=1, attention_backend=None,
    lora_name=None, lora_scale=1.0,
    lora_stack=None,
    original_image=None,
    progress=gr.Progress(track_tqdm=True),
):
    _validate("controlnet", prompt)
    if control_image is None:
        raise gr.Error("Control image is required.")

    import tempfile
    from PIL import Image as PILImage
    import numpy as np

    if isinstance(control_image, np.ndarray):
        img = PILImage.fromarray(control_image)
    else:
        img = control_image
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
    img.save(tmp.name)
    tmp.close()

    # Save original image for harmony (provides inpaint_latent reference)
    original_path = None
    if original_image is not None:
        if isinstance(original_image, np.ndarray):
            orig_img = PILImage.fromarray(original_image)
        else:
            orig_img = original_image
        tmp_orig = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
        orig_img.save(tmp_orig.name)
        tmp_orig.close()
        original_path = tmp_orig.name

    w, h = resolution.split("x")
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "control_image_path": tmp.name,
        "original_image_path": original_path,
        "control_mode": control_mode,
        "control_scale": float(control_scale),
        "width": int(w), "height": int(h),
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "cfg_normalization": bool(cfg_normalization),
        "cfg_truncation": float(cfg_truncation),
        "num_images": int(num_images),
        "max_sequence_length": int(max_sequence_length),
        "time_shift": float(time_shift),
        "seed": int(seed),
    }
    if lora_stack:
        kwargs["lora_stack"] = lora_stack
    elif lora_name:
        kwargs["lora_stack"] = [{"name": lora_name, "scale": float(lora_scale)}]
    if attention_backend:
        kwargs["attention_backend"] = attention_backend
    return _submit_and_wait("controlnet", kwargs, progress, need_controlnet=True)


# ---------------------------------------------------------------------------
# Generation: Inpaint
# ---------------------------------------------------------------------------
def generate_inpaint(
    prompt, editor_value, resolution, seed,
    negative_prompt="", num_steps=8, guidance_scale=1.0,
    cfg_truncation=1.0, control_scale=0.7,
    max_sequence_length=512, time_shift=3.0,
    lora_name=None, lora_scale=1.0,
    lora_stack=None,
    need_controlnet=True,
    step_cutoff=0.5,
    denoise=1.0,
    mask_grow=15,
    mask_blur=14,
    crop_stitch=False,
    progress=gr.Progress(track_tqdm=True),
):
    _validate("inpaint", prompt)
    if editor_value is None:
        raise gr.Error("Image with mask is required.")

    import tempfile
    from PIL import Image as PILImage
    import numpy as np
    import cv2

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
        # Grow mask then blur edges — ComfyUI best practice
        grow_px = int(mask_grow)
        blur_radius = int(mask_blur)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
        # GaussianBlur needs odd kernel size
        blur_k = blur_radius * 2 + 1
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        logger.info("Mask processed: grow=%dpx, blur_radius=%d", grow_px, blur_radius)

        image = PILImage.fromarray(background)
        mask_img = PILImage.fromarray(mask)
    else:
        raise gr.Error("Unexpected editor format.")

    # --- Crop & Stitch: crop to mask bounding box before sending to pipeline ---
    mask_np = np.array(mask_img)
    crop_info = None

    if crop_stitch:
        extend_factor = 1.2

        # Find bounding box of non-zero mask pixels
        nonzero_rows = np.any(mask_np > 0, axis=1)
        nonzero_cols = np.any(mask_np > 0, axis=0)
        if nonzero_rows.any() and nonzero_cols.any():
            r_min, r_max = np.where(nonzero_rows)[0][[0, -1]]
            c_min, c_max = np.where(nonzero_cols)[0][[0, -1]]

            # Expand bounding box by extend_factor
            bbox_h = r_max - r_min + 1
            bbox_w = c_max - c_min + 1
            expand_h = int(bbox_h * (extend_factor - 1) / 2)
            expand_w = int(bbox_w * (extend_factor - 1) / 2)

            img_h, img_w = mask_np.shape[:2]
            crop_y1 = max(0, r_min - expand_h)
            crop_y2 = min(img_h, r_max + 1 + expand_h)
            crop_x1 = max(0, c_min - expand_w)
            crop_x2 = min(img_w, c_max + 1 + expand_w)

            # Only crop if the crop region is meaningfully smaller than the full image
            crop_area = (crop_y2 - crop_y1) * (crop_x2 - crop_x1)
            full_area = img_h * img_w
            if crop_area < full_area * 0.85:
                crop_info = {
                    "x1": crop_x1, "y1": crop_y1,
                    "x2": crop_x2, "y2": crop_y2,
                    "orig_image": image.copy(),
                    "orig_w": img_w, "orig_h": img_h,
                }
                image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                mask_img = mask_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                logger.info("Crop & Stitch: cropped to (%d,%d)-(%d,%d) from %dx%d",
                            crop_x1, crop_y1, crop_x2, crop_y2, img_w, img_h)

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
        "need_controlnet": bool(need_controlnet),
        "control_step_cutoff": float(step_cutoff),
        "denoise": float(denoise),
    }
    if lora_stack:
        kwargs["lora_stack"] = lora_stack
    elif lora_name:
        kwargs["lora_stack"] = [{"name": lora_name, "scale": float(lora_scale)}]
    result = _submit_and_wait("inpaint", kwargs, progress,
                              need_controlnet=bool(need_controlnet))

    # --- Crop & Stitch: paste result back into original image ---
    # ComfyUI standard: feathered alpha blending with wide stitch mask blur
    if crop_info is not None and result is not None:
        paths, info = result
        if paths and len(paths) > 0:
            result_img = PILImage.open(paths[0]).convert("RGB")
            orig_image = crop_info["orig_image"]
            x1, y1 = crop_info["x1"], crop_info["y1"]
            x2, y2 = crop_info["x2"], crop_info["y2"]
            crop_w, crop_h = x2 - x1, y2 - y1

            # Resize result to match crop region
            result_resized = np.array(
                result_img.resize((crop_w, crop_h), PILImage.LANCZOS)
            )

            # Stitch mask: grow 24px + blur 48px (ComfyUI standard: wide feather)
            raw_mask = np.array(
                mask_img if mask_img.size == (crop_w, crop_h)
                else PILImage.fromarray(mask_np[y1:y2, x1:x2])
            )
            stitch_grow = 24
            stitch_blur = 48
            grow_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * stitch_grow + 1, 2 * stitch_grow + 1)
            )
            stitch_mask = cv2.dilate(raw_mask, grow_k, iterations=1)
            blur_k = stitch_blur * 2 + 1
            alpha = cv2.GaussianBlur(
                stitch_mask.astype(np.float32) / 255.0,
                (blur_k, blur_k), 0
            )[:, :, np.newaxis]

            original_crop = np.array(orig_image.crop((x1, y1, x2, y2)))

            # Feathered alpha blend (ComfyUI standard)
            blended = np.clip(
                alpha * result_resized.astype(np.float32)
                + (1 - alpha) * original_crop.astype(np.float32),
                0, 255
            ).astype(np.uint8)
            logger.info("Crop & Stitch: feathered alpha blend (grow=%d, blur=%d)",
                        stitch_grow, stitch_blur)

            # Paste back into full original image
            final = orig_image.copy()
            final.paste(PILImage.fromarray(blended), (x1, y1))
            final.save(paths[0])
            logger.info("Crop & Stitch: composited into full image")

    return result


# ---------------------------------------------------------------------------
# Generation: Outpaint
# ---------------------------------------------------------------------------
def generate_outpaint(
    prompt, image, direction, expand_px, resolution, seed,
    negative_prompt="", num_steps=15, guidance_scale=1.0,
    cfg_truncation=1.0, control_scale=0.5,
    max_sequence_length=512, time_shift=3.0,
    lora_name=None, lora_scale=1.0,
    lora_stack=None,
    need_controlnet=True,
    step_cutoff=0.7,
    mask_grow=120,
    mask_blur=90,
    denoise=1.0,
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
        "need_controlnet": bool(need_controlnet),
        "control_step_cutoff": float(step_cutoff),
        "mask_grow": int(mask_grow),
        "mask_blur": int(mask_blur),
        "denoise": float(denoise),
    }
    if lora_stack:
        kwargs["lora_stack"] = lora_stack
    elif lora_name:
        kwargs["lora_stack"] = [{"name": lora_name, "scale": float(lora_scale)}]
    return _submit_and_wait("outpaint", kwargs, progress,
                            need_controlnet=bool(need_controlnet))


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
