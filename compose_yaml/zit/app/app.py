"""ZIT Gradio Web UI — Z-Image-Turbo image generation."""

import argparse
import atexit
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "ui"))

import gradio as gr

from generators import (
    generate_zit_t2i,
    generate_controlnet,
    generate_inpaint,
    generate_outpaint,
    match_image_resolution,
    preview_preprocessor,
    get_gen_info_for_tab,
    get_loading_status,
    get_worker_mgr,
    set_model_dir,
)
from i18n import LANGUAGES, get_i18n_js
from pipeline_manager import OUTPUT_DIR, scan_lora_files
from translator import LANG_CHOICES as _translate_choices, DEFAULT_LANG as _translate_default


def _lora_choices():
    """LoRA dropdown choices: None + available files."""
    return ["None"] + scan_lora_files()
from zit_config import (
    MODEL_DIR,
    RESOLUTION_CHOICES,
    SAMPLE_PROMPTS,
    DEFAULT_STEPS,
    DEFAULT_TIME_SHIFT,
    DEFAULT_GUIDANCE,
    DEFAULT_CFG_TRUNCATION,
    DEFAULT_MAX_SEQ_LENGTH,
    CONTROL_MODES,
    DEFAULT_INPAINT_STEPS,
    DEFAULT_INPAINT_GUIDANCE,
    DEFAULT_INPAINT_CFG_TRUNCATION,
    DEFAULT_INPAINT_CONTROL_SCALE,
    DATASETS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zit-ui")

ATTENTION_BACKENDS = ["native", "flash", "flash_varlen", "_native_flash", "_native_math"]


def _do_kill():
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


def create_output_column(gen_type: str):
    image = gr.Image(label="Generated Image", type="filepath", buttons=["download", "fullscreen"])
    info = gr.Textbox(label="Info", interactive=False,
                      value=lambda: get_gen_info_for_tab(gen_type), every=2)
    with gr.Row():
        kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm")
    kill_msg = gr.Textbox(label="", interactive=False, visible=False)
    status_md = gr.Markdown(value=get_loading_status, every=1)
    kill_btn.click(fn=_do_kill, outputs=[kill_msg])
    return image, info


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------
def _list_outputs():
    out_dir = Path(OUTPUT_DIR)
    if not out_dir.exists():
        return []
    files = sorted(
        (f for f in out_dir.glob("*.png") if not f.name.startswith("tmp")),
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    return [str(f) for f in files[:50]]


def _get_file_info(file_path):
    if not file_path:
        return ""
    p = Path(file_path)
    json_path = p.with_suffix(".json")
    if json_path.exists():
        import json
        data = json.loads(json_path.read_text())
        return json.dumps(data, indent=2, ensure_ascii=False)
    return f"File: {p.name}\nSize: {p.stat().st_size / 1024:.0f} KB"


def _delete_selected(file_path, sel_idx):
    if not file_path:
        outputs = _list_outputs()
        if outputs:
            return gr.Gallery(value=outputs, selected_index=0), outputs[0], sel_idx, _get_file_info(outputs[0])
        return gr.Gallery(value=[]), "", 0, ""
    # Find index of the file being deleted before removal
    old_outputs = _list_outputs()
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
    outputs = _list_outputs()
    if not outputs:
        return gr.Gallery(value=[]), "", 0, ""
    # Select the image at the same position (or last if we deleted the last one)
    new_idx = min(del_idx if del_idx is not None else 0, len(outputs) - 1)
    new_path = outputs[new_idx]
    return gr.Gallery(value=outputs, selected_index=new_idx), new_path, new_idx, _get_file_info(new_path)


def _download_all():
    images = _list_outputs()
    if not images:
        return gr.update(value=None, visible=False)
    zip_path = Path(OUTPUT_DIR) / "_download_all.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for img in images:
            zf.write(img, Path(img).name)
    return gr.update(value=str(zip_path), visible=True)


def _delete_all():
    out_dir = Path(OUTPUT_DIR)
    for f in out_dir.glob("*.png"):
        f.unlink()
    for f in out_dir.glob("*.json"):
        f.unlink()
    return []


def _clear_cache():
    out_dir = Path(OUTPUT_DIR)
    # Remove temp files (masks, IPC images) from output dir
    for f in out_dir.glob("tmp*"):
        try:
            f.unlink()
        except Exception:
            pass
    # Remove underscore-prefixed cache files
    for f in out_dir.glob("_*"):
        try:
            f.unlink()
        except Exception:
            pass
    return "Cache cleared."


def _lora_list():
    from zit_config import LORAS_DIR
    loras_dir = Path(MODEL_DIR) / LORAS_DIR
    if not loras_dir.exists():
        return []
    return [[f.name, f"{f.stat().st_size / 1024 / 1024:.1f} MB"]
            for f in sorted(loras_dir.glob("*.safetensors"))]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
DATASETS_BASE = Path(MODEL_DIR) / DATASETS_DIR


def _scan_datasets():
    """Return list of dataset folder names under DATASETS_BASE."""
    DATASETS_BASE.mkdir(parents=True, exist_ok=True)
    return sorted(
        [d.name for d in DATASETS_BASE.iterdir() if d.is_dir()],
    )


def _dataset_choices():
    """Dropdown choices for dataset selector."""
    return _scan_datasets()


def _create_dataset(name: str):
    """Create a new empty dataset folder, return updated choices + selection."""
    name = name.strip().replace(" ", "_")
    if not name:
        return gr.update(), gr.update(), "Error: name is empty"
    ds_path = DATASETS_BASE / name
    ds_path.mkdir(parents=True, exist_ok=True)
    choices = _dataset_choices()
    return gr.update(choices=choices, value=name), "", f"Created: {ds_path}"


def _upload_to_dataset(files, dataset_name: str):
    """Auto-copy uploaded files into the selected dataset folder.

    Called on gr.File.upload event — copies files immediately and clears
    the drop zone so it stays ready for more files.
    Handles duplicate filenames by appending _1, _2, etc.
    """
    if not dataset_name:
        gallery, summary = _dataset_contents(dataset_name)
        return "Error: select a dataset first", gallery, summary, None
    ds_path = DATASETS_BASE / dataset_name
    ds_path.mkdir(parents=True, exist_ok=True)
    if not files:
        gallery, summary = _dataset_contents(dataset_name)
        return "No files uploaded", gallery, summary, None
    count = 0
    for f in files:
        src = Path(f)
        dst = ds_path / src.name
        # Avoid overwriting existing files — append _1, _2, ...
        if dst.exists():
            stem, suffix = dst.stem, dst.suffix
            i = 1
            while dst.exists():
                dst = ds_path / f"{stem}_{i}{suffix}"
                i += 1
        shutil.copy2(str(src), str(dst))
        count += 1
    gallery, summary = _dataset_contents(dataset_name)
    return f"Uploaded {count} file(s) to {dataset_name}/", gallery, summary, None


def _delete_dataset_image(evt: "gr.SelectData", dataset_name: str):
    """Delete a single image (and its caption) from the dataset."""
    if not dataset_name:
        return [], "No dataset selected", ""
    ds_path = DATASETS_BASE / dataset_name
    gallery, _ = _dataset_contents(dataset_name)
    idx = evt.index
    if not isinstance(idx, int) or idx < 0 or idx >= len(gallery):
        return [g for g in gallery], _, ""
    img_path = Path(gallery[idx][0])
    deleted_name = img_path.name
    if img_path.exists():
        img_path.unlink()
    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        txt_path.unlink()
    gallery, summary = _dataset_contents(dataset_name)
    return gallery, summary, f"Deleted: {deleted_name}"


def _dataset_contents(dataset_name: str):
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
    # Gallery: (filepath, caption) pairs — caption from matching .txt if exists
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
# Examples helpers
# ---------------------------------------------------------------------------
PRESETS_DIR = Path(__file__).parent / "presets"


def _list_presets():
    """List preset images that have matching JSON metadata."""
    try:
        if not PRESETS_DIR.exists():
            return []
        files = sorted(f for f in PRESETS_DIR.glob("*.png") if f.with_suffix(".json").exists())
        return [str(f) for f in files]
    except Exception:
        return []


def _load_preset_params(evt: gr.SelectData):
    """Load preset parameters from JSON when user clicks a preset image."""
    try:
        idx = evt.index if isinstance(evt.index, int) else -1
        examples = _list_presets()
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


def _delete_preset(selected_idx):
    """Delete preset image (.png) and its metadata (.json) by index."""
    try:
        if selected_idx is None or selected_idx < 0:
            return _list_presets(), "No preset selected."
        presets = _list_presets()
        if selected_idx >= len(presets):
            return _list_presets(), "Invalid preset index."
        path = Path(presets[selected_idx])
        json_path = path.with_suffix(".json")
        deleted = []
        if path.exists():
            path.unlink()
            deleted.append(path.name)
        if json_path.exists():
            json_path.unlink()
            deleted.append(json_path.name)
        return _list_presets(), f"Deleted: {', '.join(deleted)}" if deleted else (_list_presets(), "Files not found.")
    except Exception as e:
        return _list_presets(), f"Error: {e}"


def _save_as_preset(gen_paths, prompt, neg, resolution, seed,
                     steps, time_shift, cfg, cfg_norm, cfg_trunc, max_seq):
    """Save current generation as a preset."""
    try:
        if not gen_paths:
            return _list_presets(), "No image to save."
        src = Path(gen_paths[0])
        if not src.exists():
            return _list_presets(), "Image file not found."
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
        return _list_presets(), f"Saved: {name}.png"
    except Exception as e:
        return _list_presets(), f"Error: {e}"


def _export_preset(prompt, neg, resolution, seed,
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


def _import_preset(file):
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


def _extract_gallery_path(evt: "gr.SelectData"):
    """Extract original file path from Gallery select event using index.

    Gradio 6.9 caches gallery images under /tmp/gradio/, so evt.value
    contains a cache path, not the original.  We use evt.index to look up
    the original path from the current output listing instead.
    """
    try:
        outputs = _list_outputs()
        idx = evt.index
        if isinstance(idx, int) and 0 <= idx < len(outputs):
            return outputs[idx]
    except Exception:
        pass
    # Fallback: try value-based extraction
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
# UI Builder
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    def get_memory_status():
        import psutil
        vm = psutil.virtual_memory()
        used = vm.used / 1024**3
        total = vm.total / 1024**3
        return f"Memory: **{used:.1f}GB/{total:.0f}GB** ({vm.percent:.0f}% used)"

    custom_css = """
.memory-status { text-align: right; }
#gen-gallery .grid-container,
#cn-gallery .grid-container,
#history-gallery .grid-container,
#presets-gallery .grid-container,
#dataset-gallery .grid-container {
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)) !important;
}
#gen-gallery .thumbnails button,
#cn-gallery .thumbnails button,
#history-gallery .thumbnails button,
#presets-gallery .thumbnails button,
#dataset-gallery .thumbnails button {
  max-height: 200px;
  max-width: 200px;
}
#gen-gallery .thumbnails button img,
#cn-gallery .thumbnails button img,
#history-gallery .thumbnails button img,
#presets-gallery .thumbnails button img,
#dataset-gallery .thumbnails button img {
  max-height: 180px;
  object-fit: contain;
}
#history-gallery { min-height: 400px; }
@media (min-width: 769px) {
  #history-gallery { height: calc(100vh - 260px) !important; overflow-y: auto; }
}
@media (max-width: 768px) {
  #history-gallery .thumbnails { grid-template-columns: repeat(2, 1fr) !important; }
}
#presets-section .gallery { transition: max-height 0.3s ease; }
#presets-toggle-row { margin-bottom: 4px; }
#presets-toggle-row button { min-width: 80px !important; }
"""
    with gr.Blocks(title="ZIT Gradio", css=custom_css, js=get_i18n_js()) as app:
        with gr.Row():
            gr.Markdown("# ZIT Gradio")
            gr.Markdown(value=get_memory_status, every=3, elem_classes=["memory-status"])

        with gr.Tabs() as tabs:
            # ==============================================================
            # Tab 1: Generate
            # ==============================================================
            with gr.Tab("Generate", id="generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        g_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your image...")
                        with gr.Row():
                            for i, sp in enumerate(SAMPLE_PROMPTS[:3]):
                                gr.Button(f"Sample {i+1}", size="sm", min_width=60).click(
                                    fn=lambda s=sp: s, outputs=[g_prompt])
                        with gr.Accordion("Translate", open=False):
                            g_translate_target = gr.Radio(
                                ["Prompt", "Negative"], value="Prompt", label="Source", type="value",
                            )
                            with gr.Row():
                                g_translate_lang = gr.Dropdown(
                                    choices=_translate_choices, value=_translate_default,
                                    label="Target", scale=2, min_width=160,
                                )
                                g_translate_btn = gr.Button("Translate", size="sm", variant="secondary", scale=1)
                                g_translate_use = gr.Button("Use", size="sm", variant="secondary", scale=1)
                            g_translate_result = gr.Textbox(label="Translation", lines=3, interactive=False)
                        g_neg = gr.Textbox(label="Negative Prompt", lines=2)
                        g_resolution = gr.Dropdown(
                            RESOLUTION_CHOICES, value="512x768",
                            label="Resolution (WxH)", allow_custom_value=True,
                        )
                        with gr.Row():
                            g_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                            g_num = gr.Number(value=1, label="Num Images", precision=0, minimum=1, maximum=4)
                        g_steps = gr.Slider(1, 100, value=DEFAULT_STEPS, step=1, label="Steps")
                        g_time_shift = gr.Slider(1.0, 12.0, value=DEFAULT_TIME_SHIFT, step=0.5, label="Time Shift")
                        g_cfg = gr.Slider(0.0, 10.0, value=DEFAULT_GUIDANCE, step=0.5, label="Guidance Scale")
                        g_cfg_norm = gr.Checkbox(label="CFG Normalization", value=False)
                        g_cfg_trunc = gr.Slider(0.0, 1.0, value=DEFAULT_CFG_TRUNCATION, step=0.05, label="CFG Truncation")
                        g_max_seq = gr.Slider(64, 1024, value=DEFAULT_MAX_SEQ_LENGTH, step=64, label="Max Sequence Length")
                        g_use_fp8 = gr.Checkbox(label="FP8 Precision", value=True,
                            info="FP8: fast+low VRAM / OFF: BF16 original quality (reload required)")
                        g_attn = gr.Dropdown(
                            ATTENTION_BACKENDS, value="native",
                            label="Attention Backend",
                            info="native=SDPA(auto FA2), flash=FA2, _native_flash=force SDPA flash",
                        )
                        with gr.Accordion("LoRA", open=False):
                            g_lora = gr.Dropdown(
                                _lora_choices(), value="None", label="LoRA",
                                allow_custom_value=False,
                            )
                            g_lora_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="LoRA Scale")
                            g_lora_refresh = gr.Button("Refresh", size="sm", variant="secondary")
                            g_lora_refresh.click(
                                fn=lambda: gr.Dropdown(choices=_lora_choices(), value="None"),
                                outputs=[g_lora],
                            )
                        g_generate = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        with gr.Accordion("Presets", open=False, elem_id="presets-section"):
                            preset_height = gr.State(200)
                            preset_sel_idx = gr.State(-1)
                            with gr.Row(elem_id="presets-toggle-row"):
                                g_preset_expand = gr.Button("Expand", size="sm", variant="secondary")
                                g_save_preset = gr.Button("Save as Preset", size="sm", variant="primary")
                            preset_gallery = gr.Gallery(
                                label="Click to load preset",
                                value=_list_presets,
                                columns=3, height=200, object_fit="contain",
                                preview=False, elem_id="presets-gallery",
                            )
                            g_save_status = gr.Textbox(label="", interactive=False, lines=1, show_label=False)
                            g_delete_preset = gr.Button("Delete Selected Preset", size="sm", variant="stop")
                            gr.Markdown("### JSON")
                            with gr.Row():
                                g_preset_export = gr.Button("Export JSON", size="sm", variant="secondary")
                                g_preset_import = gr.UploadButton("Import JSON", size="sm", variant="secondary", file_types=[".json"])
                            g_preset_download = gr.File(visible=False)

                            def _toggle_preset_height(current_h):
                                if current_h <= 200:
                                    return 600, gr.Gallery(height=600), gr.Button(value="Collapse")
                                else:
                                    return 200, gr.Gallery(height=200), gr.Button(value="Expand")
                            g_preset_expand.click(
                                fn=_toggle_preset_height,
                                inputs=[preset_height],
                                outputs=[preset_height, preset_gallery, g_preset_expand],
                            )
                        g_gallery = gr.Gallery(label="Generated Images", columns=2, height=500, object_fit="contain", elem_id="gen-gallery", preview=True, selected_index=0)
                        g_info = gr.Textbox(label="Info", interactive=False,
                                            value=lambda: get_gen_info_for_tab("generate"), every=2)
                        with gr.Row():
                            g_kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm")
                        g_kill_msg = gr.Textbox(label="", interactive=False, visible=False)
                        gr.Markdown(value=get_loading_status, every=1)
                        g_kill_btn.click(fn=_do_kill, outputs=[g_kill_msg])
                        g_gen_paths = gr.State([])

                # Generate dispatch — ZIT only
                def _generate_dispatch(prompt, resolution, seed, num_images,
                                       neg, steps, time_shift, cfg, cfg_norm, cfg_trunc,
                                       max_seq, use_fp8, attn_backend, lora, lora_scale,
                                       progress=gr.Progress(track_tqdm=True)):
                    paths, info = generate_zit_t2i(
                        prompt, resolution, seed, num_images,
                        negative_prompt=neg, num_steps=steps,
                        time_shift=time_shift,
                        guidance_scale=cfg,
                        cfg_normalization=cfg_norm, cfg_truncation=cfg_trunc,
                        max_sequence_length=max_seq,
                        attention_backend=attn_backend,
                        lora_name=lora if lora != "None" else None,
                        lora_scale=lora_scale,
                        use_fp8=use_fp8,
                        progress=progress,
                    )
                    return gr.Gallery(value=paths, selected_index=0), info, paths

                g_generate.click(
                    fn=_generate_dispatch,
                    inputs=[g_prompt, g_resolution, g_seed, g_num,
                            g_neg, g_steps, g_time_shift, g_cfg, g_cfg_norm, g_cfg_trunc,
                            g_max_seq, g_use_fp8, g_attn, g_lora, g_lora_scale],
                    outputs=[g_gallery, g_info, g_gen_paths],
                    concurrency_limit=1,
                )

                # Presets: click to load params + track selected index
                preset_gallery.select(
                    fn=_load_preset_params,
                    outputs=[g_prompt, g_neg, g_resolution, g_seed,
                             g_steps, g_time_shift, g_cfg, g_cfg_norm, g_cfg_trunc, g_max_seq,
                             preset_sel_idx],
                )

                # Save current generation as preset
                g_save_preset.click(
                    fn=_save_as_preset,
                    inputs=[g_gen_paths, g_prompt, g_neg, g_resolution, g_seed,
                            g_steps, g_time_shift, g_cfg, g_cfg_norm, g_cfg_trunc, g_max_seq],
                    outputs=[preset_gallery, g_save_status],
                )

                # Delete selected preset (.png + .json)
                g_delete_preset.click(
                    fn=_delete_preset,
                    inputs=[preset_sel_idx],
                    outputs=[preset_gallery, g_save_status],
                )

                # Export / Import presets
                g_preset_export.click(
                    fn=_export_preset,
                    inputs=[g_prompt, g_neg, g_resolution, g_seed,
                            g_steps, g_time_shift, g_cfg, g_cfg_norm, g_cfg_trunc, g_max_seq],
                    outputs=[g_preset_download],
                )
                g_preset_import.upload(
                    fn=_import_preset,
                    inputs=[g_preset_import],
                    outputs=[g_prompt, g_neg, g_resolution, g_seed,
                             g_steps, g_time_shift, g_cfg, g_cfg_norm, g_cfg_trunc, g_max_seq],
                )

                # Translate prompt
                def _translate(text, target_lang):
                    try:
                        from translator import translate
                        return translate(text, target_lang)
                    except Exception as e:
                        return f"Error: {e}"

                def _g_translate(prompt, neg, target_sel, lang):
                    src = prompt if target_sel == "Prompt" else neg
                    return _translate(src, lang)

                def _g_use(result, prompt, neg, target_sel):
                    if target_sel == "Prompt":
                        return result, neg
                    return prompt, result

                g_translate_btn.click(
                    fn=_g_translate,
                    inputs=[g_prompt, g_neg, g_translate_target, g_translate_lang],
                    outputs=[g_translate_result],
                )
                g_translate_use.click(
                    fn=_g_use,
                    inputs=[g_translate_result, g_prompt, g_neg, g_translate_target],
                    outputs=[g_prompt, g_neg],
                )

            # ==============================================================
            # Tab 2: ControlNet
            # ==============================================================
            with gr.Tab("ControlNet", id="controlnet"):
                with gr.Row():
                    with gr.Column(scale=1):
                        cn_mode = gr.Radio(
                            CONTROL_MODES, value="canny", label="Control Mode",
                        )
                        cn_image = gr.Image(label="Input Image", type="numpy")
                        cn_preview_btn = gr.Button("Preview Preprocessor", variant="secondary", size="sm")
                        cn_preview = gr.Image(label="Control Preview", interactive=False, buttons=["download", "fullscreen"])
                        cn_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe your image...")
                        with gr.Accordion("Translate", open=False):
                            cn_translate_target = gr.Radio(
                                ["Prompt", "Negative"], value="Prompt", label="Source", type="value",
                            )
                            with gr.Row():
                                cn_translate_lang = gr.Dropdown(
                                    choices=_translate_choices, value=_translate_default,
                                    label="Target", scale=2, min_width=160,
                                )
                                cn_translate_btn = gr.Button("Translate", size="sm", variant="secondary", scale=1)
                                cn_translate_use = gr.Button("Use", size="sm", variant="secondary", scale=1)
                            cn_translate_result = gr.Textbox(label="Translation", lines=3, interactive=False)
                        cn_neg = gr.Textbox(label="Negative Prompt", lines=2)
                        cn_resolution = gr.Dropdown(
                            RESOLUTION_CHOICES, value="512x768",
                            label="Resolution (WxH)", allow_custom_value=True,
                        )
                        cn_match_res = gr.Button("Match Image Size", size="sm", variant="secondary")
                        cn_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        cn_steps = gr.Slider(1, 100, value=DEFAULT_STEPS, step=1, label="Steps")
                        cn_time_shift = gr.Slider(1.0, 12.0, value=DEFAULT_TIME_SHIFT, step=0.5, label="Time Shift")
                        cn_control_scale = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="Control Scale")
                        cn_guidance = gr.Slider(0.0, 10.0, value=DEFAULT_GUIDANCE, step=0.5, label="Guidance Scale")
                        cn_cfg_trunc = gr.Slider(0.0, 1.0, value=DEFAULT_CFG_TRUNCATION, step=0.05, label="CFG Truncation")
                        cn_max_seq = gr.Slider(64, 1024, value=DEFAULT_MAX_SEQ_LENGTH, step=64, label="Max Sequence Length")
                        cn_use_fp8 = gr.Checkbox(label="FP8 Precision", value=True,
                            info="FP8: fast+low VRAM / OFF: BF16 original quality")
                        with gr.Accordion("LoRA", open=False):
                            cn_lora = gr.Dropdown(
                                _lora_choices(), value="None", label="LoRA",
                                allow_custom_value=False,
                            )
                            cn_lora_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="LoRA Scale")
                            cn_lora_refresh = gr.Button("Refresh", size="sm", variant="secondary")
                            cn_lora_refresh.click(
                                fn=lambda: gr.Dropdown(choices=_lora_choices(), value="None"),
                                outputs=[cn_lora],
                            )
                        cn_generate = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        cn_gallery = gr.Gallery(label="Generated Images", columns=2, height=500, object_fit="contain", elem_id="cn-gallery", preview=True, selected_index=0)
                        cn_info = gr.Textbox(label="Info", interactive=False,
                                             value=lambda: get_gen_info_for_tab("controlnet"), every=2)
                        cn_kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm")
                        cn_kill_msg = gr.Textbox(label="", interactive=False, visible=False)
                        gr.Markdown(value=get_loading_status, every=1)
                        cn_kill_btn.click(fn=_do_kill, outputs=[cn_kill_msg])

                # Preview preprocessor
                cn_preview_btn.click(
                    fn=lambda img, mode: preview_preprocessor(mode, img),
                    inputs=[cn_image, cn_mode],
                    outputs=[cn_preview],
                    concurrency_limit=1,
                )

                # Match image size
                cn_match_res.click(
                    fn=match_image_resolution,
                    inputs=[cn_image],
                    outputs=[cn_resolution],
                )

                # Generate with ControlNet
                def _cn_generate(mode, prompt, neg, image, resolution, seed,
                                 steps, time_shift, control_scale, guidance, cfg_trunc, max_seq,
                                 use_fp8, lora, lora_scale,
                                 progress=gr.Progress(track_tqdm=True)):
                    # Use preview image (preprocessed) if available, else preprocess now
                    preprocessed = preview_preprocessor(mode, image)
                    paths, info = generate_controlnet(
                        prompt, mode, preprocessed, resolution, seed,
                        negative_prompt=neg, num_steps=steps, guidance_scale=guidance,
                        cfg_truncation=cfg_trunc, control_scale=control_scale,
                        max_sequence_length=max_seq, time_shift=time_shift,
                        lora_name=lora if lora != "None" else None,
                        lora_scale=lora_scale,
                        use_fp8=use_fp8,
                        progress=progress,
                    )
                    return gr.Gallery(value=paths, selected_index=0), info

                cn_generate.click(
                    fn=_cn_generate,
                    inputs=[cn_mode, cn_prompt, cn_neg, cn_image, cn_resolution, cn_seed,
                            cn_steps, cn_time_shift, cn_control_scale, cn_guidance, cn_cfg_trunc, cn_max_seq,
                            cn_use_fp8, cn_lora, cn_lora_scale],
                    outputs=[cn_gallery, cn_info],
                    concurrency_limit=1,
                )

                # Translate prompt (ControlNet)
                def _cn_translate(prompt, neg, target_sel, lang):
                    src = prompt if target_sel == "Prompt" else neg
                    return _translate(src, lang)

                def _cn_use(result, prompt, neg, target_sel):
                    if target_sel == "Prompt":
                        return result, neg
                    return prompt, result

                cn_translate_btn.click(
                    fn=_cn_translate,
                    inputs=[cn_prompt, cn_neg, cn_translate_target, cn_translate_lang],
                    outputs=[cn_translate_result],
                )
                cn_translate_use.click(
                    fn=_cn_use,
                    inputs=[cn_translate_result, cn_prompt, cn_neg, cn_translate_target],
                    outputs=[cn_prompt, cn_neg],
                )

            # ==============================================================
            # Tab 3: Inpaint/Outpaint
            # ==============================================================
            with gr.Tab("Inpaint", id="inpaint"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ip_mode = gr.Radio(["Inpaint", "Outpaint"], value="Inpaint", label="Mode")

                        # Inpaint controls
                        ip_editor = gr.ImageEditor(
                            label="Draw Mask (white = regenerate)",
                            type="numpy",
                            image_mode="RGB",
                            brush=gr.Brush(colors=["#ffffff"], default_size=20),
                            eraser=gr.Eraser(default_size=20),
                        )

                        # Outpaint controls
                        ip_out_image = gr.Image(label="Image", type="numpy", visible=False)
                        ip_direction = gr.CheckboxGroup(
                            ["Left", "Right", "Up", "Down"],
                            value=["Right"], label="Expand Direction", visible=False,
                        )
                        ip_expand = gr.Slider(64, 512, value=256, step=64, label="Expand Size (px)", visible=False)

                        ip_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe what to fill...")
                        with gr.Accordion("Translate", open=False):
                            ip_translate_target = gr.Radio(
                                ["Prompt", "Negative"], value="Prompt", label="Source", type="value",
                            )
                            with gr.Row():
                                ip_translate_lang = gr.Dropdown(
                                    choices=_translate_choices, value=_translate_default,
                                    label="Target", scale=2, min_width=160,
                                )
                                ip_translate_btn = gr.Button("Translate", size="sm", variant="secondary", scale=1)
                                ip_translate_use = gr.Button("Use", size="sm", variant="secondary", scale=1)
                            ip_translate_result = gr.Textbox(label="Translation", lines=3, interactive=False)
                        ip_neg = gr.Textbox(label="Negative Prompt", lines=2)
                        ip_resolution = gr.Dropdown(
                            RESOLUTION_CHOICES, value="512x768",
                            label="Resolution (WxH)", allow_custom_value=True,
                        )
                        ip_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        ip_steps = gr.Slider(1, 100, value=DEFAULT_INPAINT_STEPS, step=1, label="Steps")
                        ip_time_shift = gr.Slider(1.0, 12.0, value=DEFAULT_TIME_SHIFT, step=0.5, label="Time Shift")
                        ip_control_scale = gr.Slider(0.0, 1.0, value=DEFAULT_INPAINT_CONTROL_SCALE, step=0.05, label="Control Scale")
                        ip_guidance = gr.Slider(0.0, 10.0, value=DEFAULT_INPAINT_GUIDANCE, step=0.5, label="Guidance Scale")
                        ip_cfg_trunc = gr.Slider(0.0, 1.0, value=DEFAULT_INPAINT_CFG_TRUNCATION, step=0.05, label="CFG Truncation")
                        ip_max_seq = gr.Slider(64, 1024, value=DEFAULT_MAX_SEQ_LENGTH, step=64, label="Max Sequence Length")
                        ip_gen_inpaint = gr.Button("Generate", variant="primary", visible=True)
                        ip_gen_outpaint = gr.Button("Generate", variant="primary", visible=False)

                    with gr.Column(scale=1):
                        ip_result = gr.Image(label="Result", type="filepath", buttons=["download", "fullscreen"])
                        ip_info = gr.Textbox(label="Info", interactive=False,
                                             value=lambda: get_gen_info_for_tab("inpaint"), every=2)
                        ip_kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm")
                        ip_kill_msg = gr.Textbox(label="", interactive=False, visible=False)
                        gr.Markdown(value=get_loading_status, every=1)
                        ip_kill_btn.click(fn=_do_kill, outputs=[ip_kill_msg])

                # Fix: force RGB conversion on image upload (Gradio 6.9 white canvas bug)
                def _fix_editor_upload(editor_val):
                    import numpy as np
                    if editor_val is None:
                        return editor_val
                    if isinstance(editor_val, dict):
                        bg = editor_val.get("background")
                        if bg is not None and isinstance(bg, np.ndarray):
                            if bg.ndim == 3 and bg.shape[2] == 4:
                                # RGBA → RGB: composite alpha onto white background
                                alpha = bg[:, :, 3:4].astype(np.float32) / 255.0
                                rgb = bg[:, :, :3].astype(np.float32)
                                white = np.full_like(rgb, 255.0)
                                bg = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
                                editor_val["background"] = bg
                            elif bg.ndim == 2:
                                # Grayscale → RGB
                                editor_val["background"] = np.stack([bg]*3, axis=-1)
                    return editor_val

                ip_editor.upload(
                    fn=_fix_editor_upload, inputs=[ip_editor], outputs=[ip_editor],
                )

                # Mode switch: show/hide inpaint vs outpaint controls
                def _on_ip_mode(mode):
                    is_inpaint = mode == "Inpaint"
                    return [
                        gr.ImageEditor(visible=is_inpaint),
                        gr.Image(visible=not is_inpaint),
                        gr.CheckboxGroup(visible=not is_inpaint),
                        gr.Slider(visible=not is_inpaint),
                        gr.Button(visible=is_inpaint),
                        gr.Button(visible=not is_inpaint),
                    ]

                ip_mode.change(
                    fn=_on_ip_mode, inputs=[ip_mode],
                    outputs=[ip_editor, ip_out_image, ip_direction, ip_expand,
                             ip_gen_inpaint, ip_gen_outpaint],
                )

                # Translate for inpaint
                def _ip_translate(prompt, neg, target_sel, lang):
                    src = prompt if target_sel == "Prompt" else neg
                    return _translate(src, lang)

                def _ip_use(result, prompt, neg, target_sel):
                    if target_sel == "Prompt":
                        return result, neg
                    return prompt, result

                ip_translate_btn.click(
                    fn=_ip_translate,
                    inputs=[ip_prompt, ip_neg, ip_translate_target, ip_translate_lang],
                    outputs=[ip_translate_result],
                )
                ip_translate_use.click(
                    fn=_ip_use,
                    inputs=[ip_translate_result, ip_prompt, ip_neg, ip_translate_target],
                    outputs=[ip_prompt, ip_neg],
                )

                # Generate inpaint
                def _do_inpaint(editor_val, prompt, neg, resolution, seed,
                                steps, time_shift, control_scale, guidance, cfg_trunc, max_seq,
                                progress=gr.Progress(track_tqdm=True)):
                    paths, info = generate_inpaint(
                        prompt, editor_val, resolution, seed,
                        negative_prompt=neg, num_steps=steps, guidance_scale=guidance,
                        cfg_truncation=cfg_trunc, control_scale=control_scale,
                        max_sequence_length=max_seq, time_shift=time_shift,
                        progress=progress,
                    )
                    return paths[0] if paths else None, info

                ip_gen_inpaint.click(
                    fn=_do_inpaint,
                    inputs=[ip_editor,
                            ip_prompt, ip_neg, ip_resolution, ip_seed,
                            ip_steps, ip_time_shift, ip_control_scale, ip_guidance, ip_cfg_trunc, ip_max_seq],
                    outputs=[ip_result, ip_info],
                    concurrency_limit=1,
                )

                # Generate outpaint
                def _do_outpaint(out_image, direction, expand_px,
                                 prompt, neg, resolution, seed,
                                 steps, time_shift, control_scale, guidance, cfg_trunc, max_seq,
                                 progress=gr.Progress(track_tqdm=True)):
                    paths, info = generate_outpaint(
                        prompt, out_image, direction, expand_px, resolution, seed,
                        negative_prompt=neg, num_steps=steps, guidance_scale=guidance,
                        cfg_truncation=cfg_trunc, control_scale=control_scale,
                        max_sequence_length=max_seq, time_shift=time_shift,
                        progress=progress,
                    )
                    return paths[0] if paths else None, info

                ip_gen_outpaint.click(
                    fn=_do_outpaint,
                    inputs=[ip_out_image, ip_direction, ip_expand,
                            ip_prompt, ip_neg, ip_resolution, ip_seed,
                            ip_steps, ip_time_shift, ip_control_scale, ip_guidance, ip_cfg_trunc, ip_max_seq],
                    outputs=[ip_result, ip_info],
                    concurrency_limit=1,
                )

            # ==============================================================
            # Tab 4: Train LoRA
            # ==============================================================
            with gr.Tab("Train", id="train") as tr_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### LoRA Training")
                        # --- Dataset selector ---
                        _ds_choices = _dataset_choices()
                        _ds_initial = _ds_choices[0] if _ds_choices else None
                        _ds_gallery_init, _ds_summary_init = _dataset_contents(_ds_initial) if _ds_initial else ([], "No dataset selected")
                        with gr.Group():
                            tr_dataset = gr.Dropdown(
                                choices=_ds_choices,
                                value=_ds_initial,
                                label="Dataset",
                                info="Select a dataset or create a new one below",
                                allow_custom_value=False,
                            )
                            tr_ds_summary = gr.Textbox(label="Dataset", interactive=False, lines=1, show_label=False, value=_ds_summary_init)
                            tr_ds_gallery = gr.Gallery(
                                label="Dataset Images (click to delete)", columns=4, height=200,
                                object_fit="cover", preview=False,
                                elem_id="dataset-gallery",
                                value=_ds_gallery_init,
                            )
                            with gr.Accordion("Manage Dataset", open=False):
                                with gr.Row():
                                    tr_new_name = gr.Textbox(label="New Dataset Name", placeholder="my_face_dataset", scale=3)
                                    tr_create_btn = gr.Button("Create", scale=1)
                                tr_upload = gr.File(
                                    label="Drop Images & Captions here (auto-upload)",
                                    file_count="multiple",
                                    file_types=["image", ".txt"],
                                )
                                tr_ds_status = gr.Textbox(label="", interactive=False, lines=1, show_label=False)
                        # --- Training params ---
                        tr_name = gr.Textbox(label="LoRA Name", value="my_lora",
                                             info="Output: loras/<name>.safetensors")
                        with gr.Row():
                            tr_steps = gr.Number(value=2000, label="Steps", precision=0, minimum=100, maximum=50000)
                            tr_rank = gr.Dropdown([4, 8, 16, 32, 64, 128], value=16, label="Rank")
                        with gr.Row():
                            tr_lr = gr.Number(value=1e-4, label="Learning Rate")
                            tr_resolution = gr.Dropdown(
                                [256, 384, 512, 768, 1024], value=512, label="Resolution",
                            )
                        tr_batch = gr.Number(value=1, label="Batch Size", precision=0, minimum=1, maximum=4)
                        tr_grad_accum = gr.Number(value=1, label="Gradient Accumulation", precision=0, minimum=1, maximum=8)
                        tr_save_every = gr.Number(value=500, label="Save Checkpoint Every N Steps", precision=0)
                        tr_targets = gr.Textbox(
                            label="Target Modules",
                            value="to_q, to_k, to_v, to_out.0",
                            info="Comma-separated Linear layer names to train",
                        )
                        with gr.Row():
                            tr_start = gr.Button("Start Training", variant="primary")
                            tr_stop = gr.Button("Stop", variant="stop")
                    with gr.Column(scale=1):
                        tr_status = gr.Textbox(label="Status", interactive=False, lines=3)
                        tr_log = gr.Textbox(label="Training Log", interactive=False, lines=15)
                        tr_progress = gr.Markdown("Ready")

                # --- Tab select: auto-load first dataset ---
                def _on_train_tab():
                    datasets = _scan_datasets()
                    if not datasets:
                        return gr.update(choices=[], value=None), [], "No dataset selected"
                    first = datasets[0]
                    gallery, summary = _dataset_contents(first)
                    return gr.update(choices=datasets, value=first), gallery, summary

                tr_tab.select(
                    fn=_on_train_tab,
                    outputs=[tr_dataset, tr_ds_gallery, tr_ds_summary],
                )

                # --- Dataset management events ---
                tr_dataset.change(
                    fn=_dataset_contents,
                    inputs=[tr_dataset],
                    outputs=[tr_ds_gallery, tr_ds_summary],
                )
                tr_create_btn.click(
                    fn=_create_dataset,
                    inputs=[tr_new_name],
                    outputs=[tr_dataset, tr_new_name, tr_ds_status],
                )
                tr_upload.upload(
                    fn=_upload_to_dataset,
                    inputs=[tr_upload, tr_dataset],
                    outputs=[tr_ds_status, tr_ds_gallery, tr_ds_summary, tr_upload],
                )
                tr_ds_gallery.select(
                    fn=_delete_dataset_image,
                    inputs=[tr_dataset],
                    outputs=[tr_ds_gallery, tr_ds_summary, tr_ds_status],
                )

                # Train state (module-level to survive across calls)
                _trainer_ref = gr.State(None)

                def _start_training(dataset_name, name, steps, rank, lr, resolution,
                                    batch, grad_accum, save_every, targets, trainer_ref):
                    try:
                        if not dataset_name:
                            return "Error: Select a dataset", "", "Error", trainer_ref
                        dataset = str(DATASETS_BASE / dataset_name)
                        if not Path(dataset).is_dir():
                            return "Error: Dataset folder not found", "", "Error", trainer_ref
                        if not name:
                            return "Error: LoRA name required", "", "Error", trainer_ref

                        # Kill worker to free GPU
                        mgr = get_worker_mgr()
                        if mgr.is_alive():
                            mgr.kill()

                        from trainer import LoRATrainer
                        trainer = LoRATrainer(
                            model_dir=str(MODEL_DIR),
                            dataset_dir=dataset,
                            output_name=name,
                        )

                        target_list = [t.strip() for t in targets.split(",") if t.strip()]

                        log_lines = []
                        def on_progress(step, total, loss):
                            if step % 50 == 0 or step == 1:
                                log_lines.append(f"Step {step}/{total}  loss={loss:.4f}")

                        trainer.progress_callback = on_progress

                        output_path = trainer.train(
                            steps=int(steps),
                            lr=float(lr),
                            rank=int(rank),
                            batch_size=int(batch),
                            resolution=int(resolution),
                            gradient_accumulation=int(grad_accum),
                            save_every=int(save_every),
                            target_modules=target_list,
                        )

                        status = f"Training complete! Saved: {Path(output_path).name}"
                        log_text = "\n".join(log_lines[-30:])
                        return status, log_text, f"Done: {Path(output_path).name}", None

                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        return f"Error: {e}", tb, "Failed", None

                def _stop_training(trainer_ref):
                    if trainer_ref and hasattr(trainer_ref, 'stop'):
                        trainer_ref.stop()
                        return "Stop requested..."
                    return "No training in progress"

                tr_start.click(
                    fn=_start_training,
                    inputs=[tr_dataset, tr_name, tr_steps, tr_rank, tr_lr, tr_resolution,
                            tr_batch, tr_grad_accum, tr_save_every, tr_targets, _trainer_ref],
                    outputs=[tr_status, tr_log, tr_progress, _trainer_ref],
                    concurrency_limit=1,
                )
                tr_stop.click(
                    fn=_stop_training,
                    inputs=[_trainer_ref],
                    outputs=[tr_status],
                )

            # ==============================================================
            # Tab 6: Settings
            # ==============================================================
            with gr.Tab("Settings", id="settings"):
                gr.Markdown("### Language")
                with gr.Group():
                    s_lang = gr.Radio(
                        list(LANGUAGES.values()), value="English",
                        label="Language", elem_id="lang-selector",
                    )
                    s_lang.change(
                        fn=None,
                        inputs=[s_lang],
                        js="""(lang) => {
                            const map = {""" + ", ".join(f'"{v}": "{k}"' for k, v in LANGUAGES.items()) + """};
                            const code = map[lang] || 'en';
                            if (window._zit_setLang) window._zit_setLang(code);
                        }""",
                    )

                gr.Markdown("### Model Settings")
                with gr.Group():
                    s_model_dir = gr.Textbox(label="Model Directory", value=str(MODEL_DIR))
                    s_apply = gr.Button("Apply", variant="secondary", size="sm")
                    s_status = gr.Textbox(label="Status", interactive=False)

                    def _apply_model_dir(d):
                        set_model_dir(d)
                        return f"Model dir set to: {d}"
                    s_apply.click(fn=_apply_model_dir, inputs=[s_model_dir], outputs=[s_status])

                with gr.Group():
                    s_check = gr.Button("Check Models", variant="secondary", size="sm")
                    s_check_status = gr.Textbox(label="Model Status", interactive=False, lines=8)

                    def _check_models():
                        from download_models import check_status as _cs
                        import io, contextlib
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            _cs()
                        return buf.getvalue()
                    s_check.click(fn=_check_models, outputs=[s_check_status])

                gr.Markdown("### LoRA Download")
                with gr.Group():
                    s_lora_url = gr.Textbox(label="HuggingFace Repo ID or URL")
                    s_lora_fname = gr.Textbox(label="Filename in Repo (e.g. model.safetensors)")
                    s_lora_save = gr.Textbox(label="Save As (optional)")
                    s_lora_dl = gr.Button("Download", variant="secondary", size="sm")
                    s_lora_status = gr.Textbox(label="Download Status", interactive=False)

                    def _download_lora(source, fname, save_as):
                        from zit_config import LORAS_DIR
                        loras_dir = Path(MODEL_DIR) / LORAS_DIR
                        loras_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            if source.startswith("http"):
                                import urllib.request
                                out_name = save_as or source.split("/")[-1]
                                dest = loras_dir / out_name
                                urllib.request.urlretrieve(source, str(dest))
                                return f"Downloaded: {dest.name}"
                            else:
                                from huggingface_hub import hf_hub_download
                                filename = fname or "model.safetensors"
                                out_name = save_as or filename
                                hf_hub_download(source, filename, local_dir=str(loras_dir))
                                if fname and save_as and fname != save_as:
                                    (loras_dir / fname).rename(loras_dir / save_as)
                                return f"Downloaded: {out_name}"
                        except Exception as e:
                            return f"Error: {e}"

                    s_lora_dl.click(
                        fn=_download_lora,
                        inputs=[s_lora_url, s_lora_fname, s_lora_save],
                        outputs=[s_lora_status],
                    )

                gr.Markdown("### Installed LoRAs")
                with gr.Group():
                    s_lora_table = gr.Dataframe(
                        headers=["Filename", "Size"],
                        value=lambda: _lora_list(),
                        interactive=False, every=10,
                        max_height=200,
                    )
                    with gr.Row():
                        s_lora_selected = gr.Textbox(label="Selected", interactive=False, scale=3)
                        s_lora_del_btn = gr.Button("Delete Selected", variant="stop", size="sm", scale=1)
                    s_lora_del_status = gr.Textbox(label="", interactive=False, visible=False)

                    def _on_lora_select(evt: gr.SelectData):
                        loras = _lora_list()
                        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                        if isinstance(idx, int) and 0 <= idx < len(loras):
                            return loras[idx][0]
                        return ""

                    def _delete_lora(filename):
                        try:
                            if not filename or not filename.strip():
                                return _lora_list(), "", "No file selected."
                            filename = filename.strip()
                            from zit_config import LORAS_DIR
                            lora_path = Path(MODEL_DIR) / LORAS_DIR / filename
                            if not lora_path.exists():
                                return _lora_list(), "", f"Not found: {filename}"
                            lora_path.unlink()
                            return _lora_list(), "", f"Deleted: {filename}"
                        except Exception as e:
                            return _lora_list(), "", f"Error: {e}"

                    s_lora_table.select(fn=_on_lora_select, outputs=[s_lora_selected])
                    s_lora_del_btn.click(
                        fn=_delete_lora,
                        inputs=[s_lora_selected],
                        outputs=[s_lora_table, s_lora_selected, s_lora_del_status],
                    )

            # ==============================================================
            # Tab 3: History
            # ==============================================================
            with gr.Tab("History", id="history") as h_tab:
                gr.Markdown("### Generation History")
                with gr.Row():
                    h_refresh = gr.Button("Refresh", size="sm")
                    h_download_all = gr.Button("Download All", size="sm", variant="secondary")
                    h_delete = gr.Button("Delete Selected", size="sm", variant="stop")
                    h_delete_all = gr.Button("Delete All", size="sm", variant="stop")
                    h_clear_cache = gr.Button("Clear Cache", size="sm")

                with gr.Row():
                    with gr.Column(scale=3):
                        h_gallery = gr.Gallery(
                            label="Generated Images", value=_list_outputs,
                            columns=4, height=None, object_fit="contain", every=10,
                            elem_id="history-gallery", preview=True,
                            selected_index=0,
                        )
                    with gr.Column(scale=1):
                        h_selected = gr.Textbox(label="Selected File", interactive=False, visible=False)
                        h_sel_idx = gr.State(0)
                        h_file_info = gr.Textbox(label="File Info", interactive=False, lines=12)
                        h_download_file = gr.File(label="Download", visible=False, interactive=False)
                        h_cache_msg = gr.Textbox(label="", interactive=False, visible=False)

                def _on_gallery_select(evt: gr.SelectData):
                    path = _extract_gallery_path(evt)
                    idx = evt.index if isinstance(evt.index, int) else 0
                    logger.info("Gallery select index=%r path=%r", evt.index, path)
                    if path:
                        return path, idx, _get_file_info(path)
                    return "", idx, ""

                def _on_history_tab():
                    """Auto-select first image when entering History tab."""
                    outputs = _list_outputs()
                    if outputs:
                        return outputs[0], 0, _get_file_info(outputs[0])
                    return "", 0, ""

                h_gallery.select(fn=_on_gallery_select, outputs=[h_selected, h_sel_idx, h_file_info])
                h_tab.select(fn=_on_history_tab, outputs=[h_selected, h_sel_idx, h_file_info])
                h_refresh.click(fn=_list_outputs, outputs=[h_gallery])
                h_delete.click(
                    fn=_delete_selected, inputs=[h_selected, h_sel_idx],
                    outputs=[h_gallery, h_selected, h_sel_idx, h_file_info],
                )
                h_download_all.click(fn=_download_all, outputs=[h_download_file])
                h_delete_all.click(fn=_delete_all, outputs=[h_gallery])
                h_clear_cache.click(fn=_clear_cache, outputs=[h_cache_msg])

    return app


def main():
    parser = argparse.ArgumentParser(description="ZIT Gradio UI")
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_ui()

    def _cleanup():
        mgr = get_worker_mgr()
        mgr.stop()
    atexit.register(_cleanup)

    app.queue(default_concurrency_limit=None)
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
        allowed_paths=["/root/.cache/huggingface/hub/zit/datasets"],
    )


if __name__ == "__main__":
    main()
