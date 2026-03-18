"""Shared helper functions for ZIT UI tabs."""

import json
import logging
import shutil
import zipfile
from pathlib import Path

import gradio as gr

from pipeline_manager import OUTPUT_DIR, scan_lora_files
from zit_config import (
    MODEL_DIR,
    DATASETS_DIR,
    LORAS_DIR,
    DEFAULT_STEPS,
    DEFAULT_TIME_SHIFT,
    DEFAULT_GUIDANCE,
    DEFAULT_CFG_TRUNCATION,
    DEFAULT_MAX_SEQ_LENGTH,
)

logger = logging.getLogger("zit-ui")


# ---------------------------------------------------------------------------
# Fast safetensors loader
# ---------------------------------------------------------------------------
def fast_load_file(filepath, device="cpu"):
    """Load safetensors file using fastsafetensors for faster GPU loading."""
    from fastsafetensors import fastsafe_open
    state_dict = {}
    with fastsafe_open(str(filepath), device=str(device), nogds=True) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key).clone()
    return state_dict


def fast_safe_metadata(filepath):
    """Read safetensors metadata using fastsafetensors."""
    from fastsafetensors import fastsafe_open
    with fastsafe_open(str(filepath), device="cpu", nogds=True) as f:
        meta_dict = f.metadata()
    # fastsafetensors returns {filepath: OrderedDict(...)} — unwrap
    if meta_dict:
        return dict(next(iter(meta_dict.values())))
    return {}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def lora_choices():
    """LoRA dropdown choices: None + available files."""
    return ["None"] + scan_lora_files()


def lora_list():
    """List installed LoRA files as [[name, size], ...]."""
    loras_dir = Path(MODEL_DIR) / LORAS_DIR
    if not loras_dir.exists():
        return []
    return [[f.name, f"{f.stat().st_size / 1024 / 1024:.1f} MB"]
            for f in sorted(loras_dir.glob("*.safetensors"))]


def do_kill():
    """Emergency stop: kill worker + unload translator."""
    from generators import get_worker_mgr
    mgr = get_worker_mgr()
    msg = mgr.kill()
    try:
        from translator import unload as unload_translator
        unload_translator()
        msg += " | Translator unloaded."
    except Exception:
        pass
    logger.info("Kill: %s", msg)
    return msg


def do_translate(text, target_lang):
    """Translate text to target language."""
    try:
        from translator import translate
        return translate(text, target_lang)
    except Exception as e:
        return f"Error: {e}"


def translate_use(result, prompt, neg, target_sel):
    """Apply translation result to prompt or negative."""
    if target_sel == "Prompt":
        return result, neg
    return prompt, result


def get_memory_status():
    """Return memory usage string for status display."""
    import psutil
    vm = psutil.virtual_memory()
    used = vm.used / 1024**3
    total = vm.total / 1024**3
    return f"Memory: **{used:.1f}GB/{total:.0f}GB** ({vm.percent:.0f}% used)"


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------
def list_outputs():
    """List PNG output files sorted by modification time (newest first)."""
    out_dir = Path(OUTPUT_DIR)
    if not out_dir.exists():
        return []
    files = sorted(
        (f for f in out_dir.glob("*.png") if not f.name.startswith("tmp")),
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    return [str(f) for f in files[:50]]


def get_file_info(file_path):
    """Read JSON metadata or basic file info for an image."""
    if not file_path:
        return ""
    p = Path(file_path)
    json_path = p.with_suffix(".json")
    if json_path.exists():
        data = json.loads(json_path.read_text())
        return json.dumps(data, indent=2, ensure_ascii=False)
    return f"File: {p.name}\nSize: {p.stat().st_size / 1024:.0f} KB"


def delete_selected(file_path, sel_idx):
    """Delete selected history image and select next."""
    if not file_path:
        outputs = list_outputs()
        if outputs:
            return gr.Gallery(value=outputs, selected_index=0), outputs[0], sel_idx, get_file_info(outputs[0])
        return gr.Gallery(value=[]), "", 0, ""
    old_outputs = list_outputs()
    del_idx = None
    for i, p_str in enumerate(old_outputs):
        if p_str == file_path:
            del_idx = i
            break
    p = Path(file_path)
    if p.exists():
        p.unlink()
    json_p = p.with_suffix(".json")
    if json_p.exists():
        json_p.unlink()
    outputs = list_outputs()
    if not outputs:
        return gr.Gallery(value=[]), "", 0, ""
    new_idx = min(del_idx if del_idx is not None else 0, len(outputs) - 1)
    new_path = outputs[new_idx]
    return gr.Gallery(value=outputs, selected_index=new_idx), new_path, new_idx, get_file_info(new_path)


def download_all():
    """Zip all output images for download."""
    images = list_outputs()
    if not images:
        return gr.update(value=None, visible=False)
    zip_path = Path(OUTPUT_DIR) / "_download_all.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for img in images:
            zf.write(img, Path(img).name)
    return gr.update(value=str(zip_path), visible=True)


def delete_all():
    """Delete all output images and metadata."""
    out_dir = Path(OUTPUT_DIR)
    for f in out_dir.glob("*.png"):
        f.unlink()
    for f in out_dir.glob("*.json"):
        f.unlink()
    return []


def clear_cache():
    """Remove temp and cache files from output directory."""
    out_dir = Path(OUTPUT_DIR)
    for f in out_dir.glob("tmp*"):
        try:
            f.unlink()
        except Exception:
            pass
    for f in out_dir.glob("_*"):
        try:
            f.unlink()
        except Exception:
            pass
    return "Cache cleared."


def extract_gallery_path(evt: "gr.SelectData"):
    """Extract original file path from Gallery select event using index."""
    try:
        outputs = list_outputs()
        idx = evt.index
        if isinstance(idx, int) and 0 <= idx < len(outputs):
            return outputs[idx]
    except Exception:
        pass
    val = evt.value
    if not val:
        return None
    if isinstance(val, dict):
        if "image" in val and isinstance(val["image"], dict):
            return val["image"].get("path")
        if "path" in val:
            return val["path"]
    if isinstance(val, str):
        return val
    return None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
DATASETS_BASE = Path(MODEL_DIR) / DATASETS_DIR


def scan_datasets():
    """Return list of dataset folder names under DATASETS_BASE."""
    DATASETS_BASE.mkdir(parents=True, exist_ok=True)
    return sorted([d.name for d in DATASETS_BASE.iterdir() if d.is_dir()])


def dataset_choices():
    """Dropdown choices for dataset selector."""
    return scan_datasets()


def create_dataset(name: str):
    """Create a new empty dataset folder, return updated choices + selection."""
    name = name.strip().replace(" ", "_")
    if not name:
        return gr.update(), gr.update(), "Error: name is empty"
    ds_path = DATASETS_BASE / name
    ds_path.mkdir(parents=True, exist_ok=True)
    choices = dataset_choices()
    return gr.update(choices=choices, value=name), "", f"Created: {ds_path}"


def upload_to_dataset(files, dataset_name: str):
    """Auto-copy uploaded files into the selected dataset folder."""
    if not dataset_name:
        gallery, summary = dataset_contents(dataset_name)
        return "Error: select a dataset first", gallery, summary, None
    ds_path = DATASETS_BASE / dataset_name
    ds_path.mkdir(parents=True, exist_ok=True)
    if not files:
        gallery, summary = dataset_contents(dataset_name)
        return "No files uploaded", gallery, summary, None
    count = 0
    for f in files:
        src = Path(f)
        dst = ds_path / src.name
        if dst.exists():
            stem, suffix = dst.stem, dst.suffix
            i = 1
            while dst.exists():
                dst = ds_path / f"{stem}_{i}{suffix}"
                i += 1
        shutil.copy2(str(src), str(dst))
        count += 1
    gallery, summary = dataset_contents(dataset_name)
    return f"Uploaded {count} file(s) to {dataset_name}/", gallery, summary, None


def select_dataset_image(evt: "gr.SelectData", dataset_name: str):
    """Gallery click -> return image path + existing caption for editing."""
    try:
        if not dataset_name:
            return "", ""
        gallery, _ = dataset_contents(dataset_name)
        idx = evt.index
        if not isinstance(idx, int) or idx < 0 or idx >= len(gallery):
            return "", ""
        img_path = Path(gallery[idx][0])
        txt_path = img_path.with_suffix(".txt")
        caption = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
        return str(img_path), caption
    except Exception as e:
        logger.error("Select image error: %s", e)
        return "", ""


def save_caption(image_path: str, caption_text: str, dataset_name: str):
    """Save caption text to .txt file alongside image."""
    try:
        if not image_path:
            gallery, summary = dataset_contents(dataset_name)
            return "No image selected", gallery, summary
        txt_path = Path(image_path).with_suffix(".txt")
        txt_path.write_text(caption_text.strip(), encoding="utf-8")
        gallery, summary = dataset_contents(dataset_name)
        return f"Saved: {Path(image_path).name}", gallery, summary
    except Exception as e:
        logger.error("Save caption error: %s", e)
        gallery, summary = dataset_contents(dataset_name)
        return f"Error: {e}", gallery, summary


def delete_single_image(image_path: str, dataset_name: str):
    """Delete selected image + its caption .txt file."""
    try:
        if not image_path:
            gallery, summary = dataset_contents(dataset_name)
            return "No image selected", gallery, summary, "", ""
        p = Path(image_path)
        name = p.name
        if p.exists():
            p.unlink()
        txt = p.with_suffix(".txt")
        if txt.exists():
            txt.unlink()
        gallery, summary = dataset_contents(dataset_name)
        return f"Deleted: {name}", gallery, summary, "", ""
    except Exception as e:
        logger.error("Delete image error: %s", e)
        gallery, summary = dataset_contents(dataset_name)
        return f"Error: {e}", gallery, summary, "", ""


def _dedupe_tokens(existing: str, new_text: str) -> list[str]:
    """Return tokens from new_text not already in existing (comma-separated)."""
    existing_tokens = {t.strip().lower() for t in existing.split(",") if t.strip()}
    return [t.strip() for t in new_text.split(",") if t.strip() and t.strip().lower() not in existing_tokens]


def batch_prepend(text: str, dataset_name: str):
    """Prepend text to all caption .txt files (creates if missing)."""
    try:
        if not dataset_name or not text.strip():
            gallery, summary = dataset_contents(dataset_name)
            return "Need dataset and text", gallery, summary
        ds_path = DATASETS_BASE / dataset_name
        img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        imgs = sorted(f for f in ds_path.iterdir() if f.suffix.lower() in img_exts)
        modified = 0
        for img in imgs:
            txt_path = img.with_suffix(".txt")
            existing = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
            new_tokens = _dedupe_tokens(existing, text)
            if new_tokens:
                prepend_str = ", ".join(new_tokens)
                combined = f"{prepend_str}, {existing}" if existing else prepend_str
                txt_path.write_text(combined, encoding="utf-8")
                modified += 1
        gallery, summary = dataset_contents(dataset_name)
        return f"Prepended to {modified}/{len(imgs)} files (duplicates skipped)", gallery, summary
    except Exception as e:
        logger.error("Batch prepend error: %s", e)
        gallery, summary = dataset_contents(dataset_name)
        return f"Error: {e}", gallery, summary


def batch_append(text: str, dataset_name: str):
    """Append text to all caption .txt files (creates if missing)."""
    try:
        if not dataset_name or not text.strip():
            gallery, summary = dataset_contents(dataset_name)
            return "Need dataset and text", gallery, summary
        ds_path = DATASETS_BASE / dataset_name
        img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        imgs = sorted(f for f in ds_path.iterdir() if f.suffix.lower() in img_exts)
        modified = 0
        for img in imgs:
            txt_path = img.with_suffix(".txt")
            existing = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
            new_tokens = _dedupe_tokens(existing, text)
            if new_tokens:
                append_str = ", ".join(new_tokens)
                combined = f"{existing}, {append_str}" if existing else append_str
                txt_path.write_text(combined, encoding="utf-8")
                modified += 1
        gallery, summary = dataset_contents(dataset_name)
        return f"Appended to {modified}/{len(imgs)} files (duplicates skipped)", gallery, summary
    except Exception as e:
        logger.error("Batch append error: %s", e)
        gallery, summary = dataset_contents(dataset_name)
        return f"Error: {e}", gallery, summary


def batch_delete_captions(dataset_name: str):
    """Delete all .txt caption files in the dataset."""
    try:
        if not dataset_name:
            gallery, summary = dataset_contents(dataset_name)
            return "No dataset selected", gallery, summary
        ds_path = DATASETS_BASE / dataset_name
        txt_files = sorted(f for f in ds_path.iterdir() if f.suffix.lower() == ".txt")
        for f in txt_files:
            f.unlink()
        gallery, summary = dataset_contents(dataset_name)
        return f"Deleted {len(txt_files)} caption file(s)", gallery, summary
    except Exception as e:
        logger.error("Batch delete captions error: %s", e)
        gallery, summary = dataset_contents(dataset_name)
        return f"Error: {e}", gallery, summary


def auto_caption(dataset_name: str, overwrite: bool, trigger_word: str = ""):
    """Generator: auto-caption all images, yielding progress."""
    try:
        if not dataset_name:
            yield "No dataset selected", [], "No dataset"
            return
        from captioner import auto_caption_dataset
        dataset_path = str(DATASETS_BASE / dataset_name)
        for status in auto_caption_dataset(dataset_path, overwrite=overwrite,
                                           trigger_word=trigger_word):
            gallery, summary = dataset_contents(dataset_name)
            yield status, gallery, summary
    except Exception as e:
        import traceback
        logger.error("Auto-caption error: %s", traceback.format_exc())
        gallery, summary = dataset_contents(dataset_name)
        yield f"Error: {e}", gallery, summary


def dataset_contents(dataset_name: str):
    """Return (gallery_list, summary_text) for selected dataset."""
    if not dataset_name:
        return [], "No dataset selected"
    ds_path = DATASETS_BASE / dataset_name
    if not ds_path.is_dir():
        return [], "Dataset not found"
    files = sorted(ds_path.iterdir())
    if not files:
        return [], "(empty)"
    img_exts = (".jpg", ".jpeg", ".png", ".webp")
    imgs = [f for f in files if f.suffix.lower() in img_exts]
    txts = [f for f in files if f.suffix.lower() == ".txt"]
    gallery = []
    for img in imgs:
        caption_file = img.with_suffix(".txt")
        if caption_file.exists():
            caption = caption_file.read_text(encoding="utf-8").strip()[:80]
        else:
            caption = img.stem
        gallery.append((str(img), caption))
    summary = f"Images: {len(imgs)}, Captions: {len(txts)}"
    return gallery, summary


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------
PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"


def list_presets():
    """List preset images that have matching JSON metadata."""
    try:
        if not PRESETS_DIR.exists():
            return []
        files = sorted(f for f in PRESETS_DIR.glob("*.png") if f.with_suffix(".json").exists())
        return [str(f) for f in files]
    except Exception:
        return []


def load_preset_params(evt: gr.SelectData):
    """Load preset parameters from JSON when user clicks a preset image."""
    try:
        idx = evt.index if isinstance(evt.index, int) else -1
        examples = list_presets()
        path = None
        if 0 <= idx < len(examples):
            path = examples[idx]
        if not path:
            return *([gr.update()] * 10), -1
        json_path = Path(path).with_suffix(".json")
        if not json_path.exists():
            return *([gr.update()] * 10), -1
        data = json.loads(json_path.read_text())
        kw = data.get("kwargs", data)
        w = kw.get("width", 512)
        h = kw.get("height", 768)
        return [
            kw.get("prompt", ""),
            kw.get("negative_prompt", "") or "",
            f"{w}x{h}",
            kw.get("seed", -1),
            kw.get("num_steps", DEFAULT_STEPS),
            kw.get("time_shift", DEFAULT_TIME_SHIFT),
            kw.get("guidance_scale", DEFAULT_GUIDANCE),
            kw.get("cfg_normalization", False),
            kw.get("cfg_truncation", DEFAULT_CFG_TRUNCATION),
            kw.get("max_sequence_length", DEFAULT_MAX_SEQ_LENGTH),
            idx,
        ]
    except Exception:
        return *([gr.update()] * 10), -1


def delete_preset(selected_idx):
    """Delete preset image (.png) and its metadata (.json) by index."""
    try:
        if selected_idx is None or selected_idx < 0:
            return list_presets(), "No preset selected."
        presets = list_presets()
        if selected_idx >= len(presets):
            return list_presets(), "Invalid preset index."
        path = Path(presets[selected_idx])
        json_path = path.with_suffix(".json")
        deleted = []
        if path.exists():
            path.unlink()
            deleted.append(path.name)
        if json_path.exists():
            json_path.unlink()
            deleted.append(json_path.name)
        return list_presets(), f"Deleted: {', '.join(deleted)}" if deleted else (list_presets(), "Files not found.")
    except Exception as e:
        return list_presets(), f"Error: {e}"


def save_as_preset(gen_paths, prompt, neg, resolution, seed,
                   steps, time_shift, cfg, cfg_norm, cfg_trunc, max_seq):
    """Save current generation as a preset."""
    try:
        if not gen_paths:
            return list_presets(), "No image to save."
        src = Path(gen_paths[0])
        if not src.exists():
            return list_presets(), "Image file not found."
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        existing = list(PRESETS_DIR.glob("*.png"))
        idx = len(existing) + 1
        name = f"preset_{idx:03d}"
        dst = PRESETS_DIR / f"{name}.png"
        while dst.exists():
            idx += 1
            name = f"preset_{idx:03d}"
            dst = PRESETS_DIR / f"{name}.png"
        shutil.copy2(str(src), str(dst))
        w, h = (resolution.split("x") + ["768", "512"])[:2]
        meta = {
            "kwargs": {
                "prompt": prompt,
                "negative_prompt": neg or None,
                "width": int(w), "height": int(h),
                "seed": int(seed),
                "num_steps": int(steps),
                "time_shift": float(time_shift),
                "guidance_scale": float(cfg),
                "cfg_normalization": bool(cfg_norm),
                "cfg_truncation": float(cfg_trunc),
                "max_sequence_length": int(max_seq),
            }
        }
        dst.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return list_presets(), f"Saved: {name}.png"
    except Exception as e:
        return list_presets(), f"Error: {e}"


def export_preset(prompt, neg, resolution, seed,
                  steps, time_shift, cfg, cfg_norm, cfg_trunc, max_seq):
    """Export current settings as a JSON file for download."""
    try:
        w, h = (resolution.split("x") + ["768", "512"])[:2]
        meta = {
            "kwargs": {
                "prompt": prompt,
                "negative_prompt": neg or None,
                "width": int(w), "height": int(h),
                "seed": int(seed),
                "num_steps": int(steps),
                "time_shift": float(time_shift),
                "guidance_scale": float(cfg),
                "cfg_normalization": bool(cfg_norm),
                "cfg_truncation": float(cfg_trunc),
                "max_sequence_length": int(max_seq),
            }
        }
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".json", prefix="preset_", delete=False, mode="w")
        json.dump(meta, tmp, indent=2, ensure_ascii=False)
        tmp.close()
        return gr.File(value=tmp.name, visible=True)
    except Exception as e:
        logger.error("Export preset error: %s", e)
        return gr.File(value=None, visible=False)


def import_preset(file):
    """Import settings from an uploaded JSON file."""
    try:
        if file is None:
            return [gr.update()] * 10
        file_path = str(file.name) if hasattr(file, 'name') else str(file)
        data = json.loads(Path(file_path).read_text())
        kw = data.get("kwargs", data)
        w = kw.get("width", 512)
        h = kw.get("height", 768)
        return [
            kw.get("prompt", ""),
            kw.get("negative_prompt", "") or "",
            f"{w}x{h}",
            kw.get("seed", -1),
            kw.get("num_steps", DEFAULT_STEPS),
            kw.get("time_shift", DEFAULT_TIME_SHIFT),
            kw.get("guidance_scale", DEFAULT_GUIDANCE),
            kw.get("cfg_normalization", False),
            kw.get("cfg_truncation", DEFAULT_CFG_TRUNCATION),
            kw.get("max_sequence_length", DEFAULT_MAX_SEQ_LENGTH),
        ]
    except Exception:
        return [gr.update()] * 10
