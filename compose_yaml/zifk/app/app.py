"""ZIFK Gradio Web UI — Z-Image + FLUX.2 Klein unified image generation."""

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
    generate_compare,
    generate_klein_base_t2i,
    generate_klein_edit,
    generate_klein_multiref,
    generate_klein_t2i,
    generate_zimage_t2i,
    get_gen_info_for_tab,
    get_loading_status,
    get_worker_mgr,
    match_image_resolution,
    set_model_dir,
)
from i18n import LANGUAGES, get_i18n_js
from pipeline_manager import OUTPUT_DIR, scan_lora_files
from zifk_config import MODEL_DIR, RESOLUTION_CHOICES, SAMPLE_PROMPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zifk-ui")

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


def _lora_list(family: str):
    from zifk_config import LORAS_ZIMAGE_DIR, LORAS_KLEIN_DIR
    subdir = LORAS_ZIMAGE_DIR if family == "zimage" else LORAS_KLEIN_DIR
    loras_dir = Path(MODEL_DIR) / subdir
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
#gen-gallery .thumbnail-item img { max-height: 120px !important; width: auto !important; }
@media (max-width: 768px) {
  #history-gallery .thumbnails { grid-template-columns: repeat(2, 1fr) !important; }
}
"""
    with gr.Blocks(title="ZIFK", css=custom_css, js=get_i18n_js()) as app:
        with gr.Row():
            gr.Markdown("# ZIFK")
            gr.Markdown(value=get_memory_status, every=3, elem_classes=["memory-status"])

        with gr.Tabs() as tabs:
            # ==============================================================
            # Tab 1: Generate
            # ==============================================================
            with gr.Tab("Generate", id="generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        g_model = gr.Radio(
                            ["ZIT (Fast)", "ZIB (Creative)", "Klein (Distilled)", "Klein Base"],
                            value="ZIT (Fast)", label="Model",
                        )
                        g_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your image...")
                        with gr.Row():
                            for i, sp in enumerate(SAMPLE_PROMPTS[:3]):
                                gr.Button(f"Sample {i+1}", size="sm", min_width=60).click(
                                    fn=lambda s=sp: s, outputs=[g_prompt])
                        g_resolution = gr.Dropdown(
                            RESOLUTION_CHOICES, value="512x768",
                            label="Resolution (WxH)", allow_custom_value=True,
                        )
                        with gr.Row():
                            g_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                            g_num = gr.Number(value=1, label="Num Images", precision=0, minimum=1, maximum=4)

                        # Advanced: Z-Image params
                        with gr.Accordion("Z-Image Advanced", open=False, visible=True) as g_zi_adv:
                            g_neg = gr.Textbox(label="Negative Prompt", lines=2)
                            g_steps = gr.Slider(1, 100, value=8, step=1, label="Steps")
                            g_cfg = gr.Slider(0.0, 10.0, value=0.5, step=0.5, label="Guidance Scale")
                            g_cfg_norm = gr.Checkbox(label="CFG Normalization", value=False)
                            g_cfg_trunc = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="CFG Truncation")
                            g_max_seq = gr.Slider(64, 1024, value=512, step=64, label="Max Sequence Length")

                        # Advanced: Klein params
                        with gr.Accordion("Klein Advanced", open=False, visible=False) as g_kl_adv:
                            g_kl_steps = gr.Slider(1, 100, value=4, step=1, label="Steps")
                            g_kl_guidance = gr.Slider(0.0, 10.0, value=1.0, step=0.5, label="Guidance")

                        g_generate = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        g_gallery = gr.Gallery(label="Generated Images", columns=2, height=500, object_fit="contain", elem_id="gen-gallery")
                        g_info = gr.Textbox(label="Info", interactive=False,
                                            value=lambda: get_gen_info_for_tab("generate"), every=2)
                        with gr.Row():
                            g_kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm")
                            g_send_edit = gr.Button("Send to Edit", size="sm", variant="secondary")
                        g_kill_msg = gr.Textbox(label="", interactive=False, visible=False)
                        gr.Markdown(value=get_loading_status, every=1)
                        g_kill_btn.click(fn=_do_kill, outputs=[g_kill_msg])
                        g_gen_paths = gr.State([])

                # Model switch: show/hide params, update defaults
                def _on_model_change(model):
                    is_zi = model in ("ZIT (Fast)", "ZIB (Creative)")
                    is_zib = model == "ZIB (Creative)"
                    is_klein = model in ("Klein (Distilled)", "Klein Base")
                    is_klein_base = model == "Klein Base"

                    # Z-Image accordion visible for ZIT/ZIB
                    zi_vis = gr.Accordion(visible=is_zi)
                    # Klein accordion visible for Klein variants
                    kl_vis = gr.Accordion(visible=is_klein)
                    # Klein defaults
                    kl_steps = gr.Slider(value=50 if is_klein_base else 4)
                    kl_guidance = gr.Slider(value=4.0 if is_klein_base else 1.0)
                    # ZIB defaults
                    zi_steps = gr.Slider(value=8 if model == "ZIT (Fast)" else 28)
                    zi_cfg = gr.Slider(value=0.5 if model == "ZIT (Fast)" else 3.5)
                    zi_cfg_trunc = gr.Slider(value=0.9 if model == "ZIT (Fast)" else 1.0)

                    return [zi_vis, kl_vis, kl_steps, kl_guidance, zi_steps, zi_cfg, zi_cfg_trunc]

                g_model.change(
                    fn=_on_model_change, inputs=[g_model],
                    outputs=[g_zi_adv, g_kl_adv, g_kl_steps, g_kl_guidance, g_steps, g_cfg, g_cfg_trunc],
                )

                # Generate dispatch
                def _generate_dispatch(model, prompt, resolution, seed, num_images,
                                       neg, steps, cfg, cfg_norm, cfg_trunc, max_seq,
                                       kl_steps, kl_guidance,
                                       progress=gr.Progress(track_tqdm=True)):
                    if model == "Klein (Distilled)":
                        paths, info = generate_klein_t2i(
                            prompt, resolution, seed,
                            num_steps=kl_steps, guidance=kl_guidance,
                            progress=progress,
                        )
                    elif model == "Klein Base":
                        paths, info = generate_klein_base_t2i(
                            prompt, resolution, seed,
                            num_steps=kl_steps, guidance=kl_guidance,
                            progress=progress,
                        )
                    else:
                        paths, info = generate_zimage_t2i(
                            prompt, model, resolution, seed, num_images,
                            negative_prompt=neg, num_steps=steps, guidance_scale=cfg,
                            cfg_normalization=cfg_norm, cfg_truncation=cfg_trunc,
                            max_sequence_length=max_seq,
                            progress=progress,
                        )
                    return paths, info, paths

                g_generate.click(
                    fn=_generate_dispatch,
                    inputs=[g_model, g_prompt, g_resolution, g_seed, g_num,
                            g_neg, g_steps, g_cfg, g_cfg_norm, g_cfg_trunc, g_max_seq,
                            g_kl_steps, g_kl_guidance],
                    outputs=[g_gallery, g_info, g_gen_paths],
                )

            # ==============================================================
            # Tab 2: Edit (Klein)
            # ==============================================================
            with gr.Tab("Edit", id="edit"):
                with gr.Row():
                    with gr.Column(scale=1):
                        e_mode = gr.Radio(
                            ["Edit (Single Ref)", "Multi-Reference"],
                            value="Edit (Single Ref)", label="Mode",
                        )
                        e_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the edit or generation...")
                        e_image = gr.Image(label="Input Image", type="numpy")
                        e_images_gallery = gr.Gallery(
                            label="Reference Images (Multi-Ref)", columns=4, height=150, visible=False,
                        )
                        e_add_ref = gr.UploadButton("+ Add Reference", file_types=["image"], visible=False)
                        e_clear_refs = gr.Button("Clear References", size="sm", visible=False)
                        e_resolution = gr.Dropdown(
                            RESOLUTION_CHOICES, value="512x768",
                            label="Resolution (WxH)", allow_custom_value=True,
                        )
                        e_match_res = gr.Button("Match Image Size", size="sm", variant="secondary")
                        with gr.Accordion("Klein Parameters", open=False):
                            e_kl_variant = gr.Radio(
                                ["Distilled", "Base"],
                                value="Distilled", label="Klein Variant",
                            )
                            e_kl_steps = gr.Slider(1, 100, value=4, step=1, label="Steps")
                            e_kl_guidance = gr.Slider(0.0, 10.0, value=1.0, step=0.5, label="Guidance")
                        e_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        e_generate = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        e_out_image, e_info = create_output_column("edit")

                # Mode switch
                def _on_edit_mode(mode):
                    is_multi = mode == "Multi-Reference"
                    return [
                        gr.Image(visible=not is_multi),
                        gr.Gallery(visible=is_multi),
                        gr.UploadButton(visible=is_multi),
                        gr.Button(visible=is_multi),
                    ]

                e_mode.change(
                    fn=_on_edit_mode, inputs=[e_mode],
                    outputs=[e_image, e_images_gallery, e_add_ref, e_clear_refs],
                )

                # Klein variant switch: update defaults
                def _on_edit_variant(variant):
                    is_base = variant == "Base"
                    return [
                        gr.Slider(value=50 if is_base else 4),
                        gr.Slider(value=4.0 if is_base else 1.0),
                    ]

                e_kl_variant.change(
                    fn=_on_edit_variant, inputs=[e_kl_variant],
                    outputs=[e_kl_steps, e_kl_guidance],
                )

                # Match image size
                def _match_res(image):
                    return match_image_resolution(image)

                e_match_res.click(fn=_match_res, inputs=[e_image], outputs=[e_resolution])

                # Multi-ref image management
                _multiref_images = gr.State([])

                def _add_ref_image(file, current_images):
                    if file is None:
                        return current_images, current_images
                    from PIL import Image as PILImage
                    img = PILImage.open(file.name)
                    current_images = current_images or []
                    current_images.append(img)
                    return current_images, [img for img in current_images]

                def _clear_refs():
                    return [], []

                e_add_ref.upload(
                    fn=_add_ref_image,
                    inputs=[e_add_ref, _multiref_images],
                    outputs=[_multiref_images, e_images_gallery],
                )
                e_clear_refs.click(
                    fn=_clear_refs, outputs=[_multiref_images, e_images_gallery],
                )

                # Edit generate
                def _edit_dispatch(mode, prompt, image, resolution, seed,
                                   kl_variant, kl_steps, kl_guidance, multi_images,
                                   progress=gr.Progress(track_tqdm=True)):
                    variant = "flux.2-klein-base-4b" if kl_variant == "Base" else "flux.2-klein-4b"
                    if mode == "Multi-Reference":
                        if not multi_images:
                            raise gr.Error("Add reference images first.")
                        paths, info = generate_klein_multiref(
                            prompt, multi_images, resolution, seed,
                            klein_variant=variant,
                            num_steps=kl_steps, guidance=kl_guidance,
                            progress=progress,
                        )
                    else:
                        paths, info = generate_klein_edit(
                            prompt, image, resolution, seed,
                            klein_variant=variant,
                            num_steps=kl_steps, guidance=kl_guidance,
                            progress=progress,
                        )
                    return paths[0], info

                e_generate.click(
                    fn=_edit_dispatch,
                    inputs=[e_mode, e_prompt, e_image, e_resolution, e_seed,
                            e_kl_variant, e_kl_steps, e_kl_guidance, _multiref_images],
                    outputs=[e_out_image, e_info],
                )

            # ==============================================================
            # Tab 3: Compare
            # ==============================================================
            with gr.Tab("Compare", id="compare"):
                with gr.Row():
                    with gr.Column(scale=1):
                        c_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Same prompt across models...")
                        c_resolution = gr.Dropdown(
                            RESOLUTION_CHOICES, value="512x768",
                            label="Resolution (WxH)", allow_custom_value=True,
                        )
                        c_seed = gr.Number(value=42, label="Seed (fixed recommended)", precision=0)
                        with gr.Row():
                            c_zit = gr.Checkbox(label="ZIT", value=True)
                            c_zib = gr.Checkbox(label="ZIB", value=True)
                            c_klein = gr.Checkbox(label="Klein", value=True)
                            c_klein_base = gr.Checkbox(label="Klein Base", value=False)
                        with gr.Accordion("Model Parameters", open=False):
                            with gr.Row():
                                c_zit_steps = gr.Slider(1, 20, value=8, step=1, label="ZIT Steps")
                                c_zit_guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.5, label="ZIT Guidance")
                            c_neg = gr.Textbox(label="ZIB Negative Prompt", lines=2)
                            with gr.Row():
                                c_zib_steps = gr.Slider(10, 100, value=28, step=1, label="ZIB Steps")
                                c_zib_cfg = gr.Slider(1.0, 10.0, value=3.5, step=0.5, label="ZIB CFG")
                            with gr.Row():
                                c_kl_steps = gr.Slider(1, 100, value=4, step=1, label="Klein Steps")
                                c_kl_guidance = gr.Slider(0.0, 10.0, value=1.0, step=0.5, label="Klein Guidance")
                            with gr.Row():
                                c_klb_steps = gr.Slider(10, 100, value=50, step=1, label="Klein Base Steps")
                                c_klb_guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="Klein Base Guidance")
                        c_compare = gr.Button("Compare", variant="primary")

                    with gr.Column(scale=1):
                        c_gallery = gr.Gallery(label="Comparison Results", columns=4, height=500, object_fit="contain")
                        c_info = gr.Textbox(label="Info", interactive=False, lines=6)
                        kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm")
                        kill_btn.click(fn=_do_kill, outputs=[gr.Textbox(visible=False)])

                def _run_compare(prompt, resolution, seed,
                                 use_zit, use_zib, use_klein, use_klein_base,
                                 zit_steps, zit_guidance,
                                 neg, zib_steps, zib_cfg,
                                 kl_steps, kl_guidance,
                                 klb_steps, klb_guidance,
                                 progress=gr.Progress(track_tqdm=True)):
                    results = generate_compare(
                        prompt, resolution, seed, use_zit, use_zib, use_klein, use_klein_base,
                        negative_prompt=neg, zit_steps=zit_steps, zit_guidance=zit_guidance,
                        zib_steps=zib_steps, zib_cfg=zib_cfg,
                        klein_steps=kl_steps, klein_guidance=kl_guidance,
                        klein_base_steps=klb_steps, klein_base_guidance=klb_guidance,
                        progress=progress,
                    )
                    images = []
                    info_lines = []
                    for path, label in results:
                        if path:
                            images.append((path, label.split("|")[0].strip()))
                        info_lines.append(label)
                    return images, "\n".join(info_lines)

                c_compare.click(
                    fn=_run_compare,
                    inputs=[c_prompt, c_resolution, c_seed,
                            c_zit, c_zib, c_klein, c_klein_base,
                            c_zit_steps, c_zit_guidance,
                            c_neg, c_zib_steps, c_zib_cfg,
                            c_kl_steps, c_kl_guidance,
                            c_klb_steps, c_klb_guidance],
                    outputs=[c_gallery, c_info],
                )

            # ==============================================================
            # Tab 4: Settings
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
                            if (window._zifk_setLang) window._zifk_setLang(code);
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

                gr.Markdown("### Z-Image Attention Backend")
                with gr.Group():
                    s_attn = gr.Dropdown(
                        ATTENTION_BACKENDS, value="native",
                        label="Attention Backend (Z-Image only)",
                        info="native=SDPA(auto FA2), flash=FA2, _native_flash=force SDPA flash",
                    )
                    s_attn_status = gr.Textbox(label="", interactive=False, visible=False)

                gr.Markdown("### LoRA Download")
                with gr.Group():
                    s_lora_family = gr.Radio(["zimage", "klein"], value="zimage", label="Model Family")
                    s_lora_url = gr.Textbox(label="HuggingFace Repo ID or URL")
                    s_lora_fname = gr.Textbox(label="Filename in Repo (e.g. model.safetensors)")
                    s_lora_save = gr.Textbox(label="Save As (optional)")
                    s_lora_dl = gr.Button("Download", variant="secondary", size="sm")
                    s_lora_status = gr.Textbox(label="Download Status", interactive=False)

                    def _download_lora(family, source, fname, save_as):
                        from zifk_config import LORAS_ZIMAGE_DIR, LORAS_KLEIN_DIR
                        subdir = LORAS_ZIMAGE_DIR if family == "zimage" else LORAS_KLEIN_DIR
                        loras_dir = Path(MODEL_DIR) / subdir
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
                                return f"Downloaded: {out_name} -> {family}"
                        except Exception as e:
                            return f"Error: {e}"

                    s_lora_dl.click(
                        fn=_download_lora,
                        inputs=[s_lora_family, s_lora_url, s_lora_fname, s_lora_save],
                        outputs=[s_lora_status],
                    )

                gr.Markdown("### Installed LoRAs")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Z-Image LoRAs**")
                        gr.Dataframe(
                            headers=["Filename", "Size"],
                            value=lambda: _lora_list("zimage"),
                            interactive=False, every=10,
                        )
                    with gr.Column():
                        gr.Markdown("**Klein LoRAs**")
                        gr.Dataframe(
                            headers=["Filename", "Size"],
                            value=lambda: _lora_list("klein"),
                            interactive=False, every=10,
                        )

            # ==============================================================
            # Tab 5: History
            # ==============================================================
            with gr.Tab("History", id="history"):
                gr.Markdown("### Generation History")
                with gr.Row():
                    h_refresh = gr.Button("Refresh", size="sm")
                    h_delete = gr.Button("Delete Selected", size="sm", variant="stop")
                    h_delete_all = gr.Button("Delete All", size="sm", variant="stop")
                    h_clear_cache = gr.Button("Clear Cache", size="sm")

                h_gallery = gr.Gallery(
                    label="Generated Images", value=_list_outputs,
                    columns=4, height=600, object_fit="contain", every=10,
                    elem_id="history-gallery",
                )
                h_selected = gr.Textbox(label="Selected File", interactive=False, visible=False)
                h_file_info = gr.Textbox(label="File Info", interactive=False, lines=8)
                h_cache_msg = gr.Textbox(label="", interactive=False, visible=False)
                h_send_edit = gr.Button("Send to Edit", size="sm", variant="secondary")

                def _on_gallery_select(evt: gr.SelectData):
                    if evt.value and "name" in evt.value:
                        path = evt.value["name"]
                        return path, _get_file_info(path)
                    return "", ""

                h_gallery.select(fn=_on_gallery_select, outputs=[h_selected, h_file_info])
                h_refresh.click(fn=_list_outputs, outputs=[h_gallery])
                h_delete.click(fn=_delete_selected, inputs=[h_selected], outputs=[h_gallery])
                h_delete_all.click(fn=_delete_all, outputs=[h_gallery])
                h_clear_cache.click(fn=_clear_cache, outputs=[h_cache_msg])

        # ---------------------------------------------------------------
        # Send to Edit (cross-tab image transfer)
        # ---------------------------------------------------------------
        def _send_to_edit(image_path):
            if not image_path:
                return gr.Tabs(), None, "512x768"
            from PIL import Image as PILImage
            import numpy as np
            img = PILImage.open(image_path)
            res = match_image_resolution(img)
            return gr.Tabs(selected="edit"), np.array(img), res

        def _send_to_edit_from_gen(paths):
            if not paths:
                return gr.Tabs(), None, "512x768"
            return _send_to_edit(paths[0])

        g_send_edit.click(
            fn=_send_to_edit_from_gen, inputs=[g_gen_paths],
            outputs=[tabs, e_image, e_resolution],
        )
        h_send_edit.click(
            fn=_send_to_edit, inputs=[h_selected],
            outputs=[tabs, e_image, e_resolution],
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="ZIFK Gradio UI")
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_ui()

    def _cleanup():
        mgr = get_worker_mgr()
        mgr.stop()
    atexit.register(_cleanup)

    app.queue()
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
