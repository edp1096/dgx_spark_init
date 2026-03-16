"""ZIT Gradio Web UI — Z-Image-Turbo image generation."""

import argparse
import atexit
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "ui"))

import gradio as gr

from generators import (
    generate_zit_t2i,
    generate_controlnet,
    generate_inpaint,
    generate_outpaint,
    generate_faceswap,
    match_image_resolution,
    preview_preprocessor,
    get_gen_info_for_tab,
    get_loading_status,
    get_worker_mgr,
    set_model_dir,
)
from i18n import LANGUAGES, get_i18n_js
from pipeline_manager import OUTPUT_DIR, scan_lora_files
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
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zit-ui")

ATTENTION_BACKENDS = ["native", "flash", "flash_varlen", "_native_flash", "_native_math"]


def _do_kill():
    mgr = get_worker_mgr()
    msg = mgr.kill()
    logger.info("Kill: %s", msg)
    return msg


def create_output_column(gen_type: str):
    image = gr.Image(label="Generated Image", type="filepath")
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
    files = sorted(out_dir.glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
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


def _delete_selected(file_path):
    if not file_path:
        return _list_outputs()
    p = Path(file_path)
    if p.exists():
        p.unlink()
    json_p = p.with_suffix(".json")
    if json_p.exists():
        json_p.unlink()
    return _list_outputs()


def _delete_all():
    out_dir = Path(OUTPUT_DIR)
    for f in out_dir.glob("*.png"):
        f.unlink()
    for f in out_dir.glob("*.json"):
        f.unlink()
    return []


def _clear_cache():
    out_dir = Path(OUTPUT_DIR)
    for f in out_dir.glob("_*"):
        f.unlink()
    import tempfile
    tmp = Path(tempfile.gettempdir())
    for f in tmp.glob("tmp*.png"):
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
#gen-gallery .thumbnails button,
#cn-gallery .thumbnails button,
#history-gallery .thumbnails button {
  max-height: 200px;
  max-width: 200px;
}
#gen-gallery .thumbnails button img,
#cn-gallery .thumbnails button img,
#history-gallery .thumbnails button img {
  max-height: 180px;
  object-fit: contain;
}
#history-gallery { min-height: 400px; }
@media (max-width: 768px) {
  #history-gallery .thumbnails { grid-template-columns: repeat(2, 1fr) !important; }
}
"""
    with gr.Blocks(title="ZIT", css=custom_css, js=get_i18n_js()) as app:
        with gr.Row():
            gr.Markdown("# ZIT")
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
                        g_attn = gr.Dropdown(
                            ATTENTION_BACKENDS, value="native",
                            label="Attention Backend",
                            info="native=SDPA(auto FA2), flash=FA2, _native_flash=force SDPA flash",
                        )
                        g_generate = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
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
                                       max_seq, attn_backend,
                                       progress=gr.Progress(track_tqdm=True)):
                    paths, info = generate_zit_t2i(
                        prompt, resolution, seed, num_images,
                        negative_prompt=neg, num_steps=steps,
                        time_shift=time_shift,
                        guidance_scale=cfg,
                        cfg_normalization=cfg_norm, cfg_truncation=cfg_trunc,
                        max_sequence_length=max_seq,
                        attention_backend=attn_backend,
                        progress=progress,
                    )
                    return gr.Gallery(value=paths, selected_index=0), info, paths

                g_generate.click(
                    fn=_generate_dispatch,
                    inputs=[g_prompt, g_resolution, g_seed, g_num,
                            g_neg, g_steps, g_time_shift, g_cfg, g_cfg_norm, g_cfg_trunc,
                            g_max_seq, g_attn],
                    outputs=[g_gallery, g_info, g_gen_paths],
                    concurrency_limit=1,
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
                        cn_preview = gr.Image(label="Control Preview", interactive=False)
                        cn_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe your image...")
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
                                 progress=gr.Progress(track_tqdm=True)):
                    # Use preview image (preprocessed) if available, else preprocess now
                    preprocessed = preview_preprocessor(mode, image)
                    paths, info = generate_controlnet(
                        prompt, mode, preprocessed, resolution, seed,
                        negative_prompt=neg, num_steps=steps, guidance_scale=guidance,
                        cfg_truncation=cfg_trunc, control_scale=control_scale,
                        max_sequence_length=max_seq, time_shift=time_shift,
                        progress=progress,
                    )
                    return gr.Gallery(value=paths, selected_index=0), info

                cn_generate.click(
                    fn=_cn_generate,
                    inputs=[cn_mode, cn_prompt, cn_neg, cn_image, cn_resolution, cn_seed,
                            cn_steps, cn_time_shift, cn_control_scale, cn_guidance, cn_cfg_trunc, cn_max_seq],
                    outputs=[cn_gallery, cn_info],
                    concurrency_limit=1,
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
                        ip_neg = gr.Textbox(label="Negative Prompt", lines=2)
                        ip_resolution = gr.Dropdown(
                            RESOLUTION_CHOICES, value="512x768",
                            label="Resolution (WxH)", allow_custom_value=True,
                        )
                        ip_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        ip_steps = gr.Slider(1, 100, value=DEFAULT_STEPS, step=1, label="Steps")
                        ip_time_shift = gr.Slider(1.0, 12.0, value=DEFAULT_TIME_SHIFT, step=0.5, label="Time Shift")
                        ip_control_scale = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Control Scale")
                        ip_guidance = gr.Slider(0.0, 10.0, value=DEFAULT_GUIDANCE, step=0.5, label="Guidance Scale")
                        ip_cfg_trunc = gr.Slider(0.0, 1.0, value=DEFAULT_CFG_TRUNCATION, step=0.05, label="CFG Truncation")
                        ip_max_seq = gr.Slider(64, 1024, value=DEFAULT_MAX_SEQ_LENGTH, step=64, label="Max Sequence Length")
                        ip_generate = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        ip_result = gr.Image(label="Result", type="filepath")
                        ip_info = gr.Textbox(label="Info", interactive=False,
                                             value=lambda: get_gen_info_for_tab("inpaint"), every=2)
                        ip_kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm")
                        ip_kill_msg = gr.Textbox(label="", interactive=False, visible=False)
                        gr.Markdown(value=get_loading_status, every=1)
                        ip_kill_btn.click(fn=_do_kill, outputs=[ip_kill_msg])

                # Mode switch: show/hide inpaint vs outpaint controls
                def _on_ip_mode(mode):
                    is_inpaint = mode == "Inpaint"
                    return [
                        gr.ImageEditor(visible=is_inpaint),
                        gr.Image(visible=not is_inpaint),
                        gr.CheckboxGroup(visible=not is_inpaint),
                        gr.Slider(visible=not is_inpaint),
                    ]

                ip_mode.change(
                    fn=_on_ip_mode, inputs=[ip_mode],
                    outputs=[ip_editor, ip_out_image, ip_direction, ip_expand],
                )

                # Generate inpaint/outpaint
                def _ip_generate(mode, editor_val, out_image, direction, expand_px,
                                 prompt, neg, resolution, seed,
                                 steps, time_shift, control_scale, guidance, cfg_trunc, max_seq,
                                 progress=gr.Progress(track_tqdm=True)):
                    import logging as _log
                    _log.getLogger("zit-ui").info(
                        "_ip_generate called: mode=%s out_image_type=%s direction=%s",
                        mode, type(out_image).__name__, direction)
                    if mode == "Inpaint":
                        paths, info = generate_inpaint(
                            prompt, editor_val, resolution, seed,
                            negative_prompt=neg, num_steps=steps, guidance_scale=guidance,
                            cfg_truncation=cfg_trunc, control_scale=control_scale,
                            max_sequence_length=max_seq, time_shift=time_shift,
                            progress=progress,
                        )
                    else:
                        paths, info = generate_outpaint(
                            prompt, out_image, direction, expand_px, resolution, seed,
                            negative_prompt=neg, num_steps=steps, guidance_scale=guidance,
                            cfg_truncation=cfg_trunc, control_scale=control_scale,
                            max_sequence_length=max_seq, time_shift=time_shift,
                            progress=progress,
                        )
                    return paths[0] if paths else None, info

                ip_generate.click(
                    fn=_ip_generate,
                    inputs=[ip_mode, ip_editor, ip_out_image, ip_direction, ip_expand,
                            ip_prompt, ip_neg, ip_resolution, ip_seed,
                            ip_steps, ip_time_shift, ip_control_scale, ip_guidance, ip_cfg_trunc, ip_max_seq],
                    outputs=[ip_result, ip_info],
                    concurrency_limit=1,
                )

            # ==============================================================
            # Tab 4: FaceSwap
            # ==============================================================
            with gr.Tab("FaceSwap", id="faceswap"):
                with gr.Row():
                    with gr.Column(scale=1):
                        fs_target = gr.Image(label="Target Image (face to replace)", type="numpy")
                        fs_source = gr.Image(label="Source Face (reference)", type="numpy")
                        fs_swap = gr.Button("Swap Face", variant="primary")
                    with gr.Column(scale=1):
                        fs_result = gr.Image(label="Result")
                        fs_info = gr.Textbox(label="Info", interactive=False)

                def _swap_face(target, source, progress=gr.Progress(track_tqdm=True)):
                    try:
                        return generate_faceswap(target, source, progress=progress), "Face swap complete"
                    except Exception as e:
                        return None, f"Error: {e}"

                fs_swap.click(
                    fn=_swap_face,
                    inputs=[fs_target, fs_source],
                    outputs=[fs_result, fs_info],
                    concurrency_limit=1,
                )

            # ==============================================================
            # Tab 5: Settings
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
                gr.Dataframe(
                    headers=["Filename", "Size"],
                    value=lambda: _lora_list(),
                    interactive=False, every=10,
                )

            # ==============================================================
            # Tab 3: History
            # ==============================================================
            with gr.Tab("History", id="history"):
                gr.Markdown("### Generation History")
                with gr.Row():
                    h_refresh = gr.Button("Refresh", size="sm")
                    h_delete = gr.Button("Delete Selected", size="sm", variant="stop")
                    h_delete_all = gr.Button("Delete All", size="sm", variant="stop")
                    h_clear_cache = gr.Button("Clear Cache", size="sm")

                with gr.Row():
                    with gr.Column(scale=3):
                        h_gallery = gr.Gallery(
                            label="Generated Images", value=_list_outputs,
                            columns=4, height=None, object_fit="contain", every=10,
                            elem_id="history-gallery", preview=True,
                        )
                    with gr.Column(scale=1):
                        h_selected = gr.Textbox(label="Selected File", interactive=False, visible=False)
                        h_file_info = gr.Textbox(label="File Info", interactive=False, lines=12)
                        h_cache_msg = gr.Textbox(label="", interactive=False, visible=False)

                def _extract_gallery_path(evt_value):
                    """Extract file path from Gallery select event value (Gradio 6.9+)."""
                    if not evt_value:
                        return None
                    # Gradio 6.9: {"image": {"path": "..."}, "caption": "..."}
                    if isinstance(evt_value, dict):
                        if "image" in evt_value and isinstance(evt_value["image"], dict):
                            return evt_value["image"].get("path")
                        if "path" in evt_value:
                            return evt_value["path"]
                        if "name" in evt_value:
                            return evt_value["name"]
                    if isinstance(evt_value, str):
                        return evt_value
                    return None

                def _on_gallery_select(evt: gr.SelectData):
                    path = _extract_gallery_path(evt.value)
                    if path:
                        return path, _get_file_info(path)
                    return "", ""

                h_gallery.select(fn=_on_gallery_select, outputs=[h_selected, h_file_info])
                h_refresh.click(fn=_list_outputs, outputs=[h_gallery])
                h_delete.click(fn=_delete_selected, inputs=[h_selected], outputs=[h_gallery])
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
    )


if __name__ == "__main__":
    main()
