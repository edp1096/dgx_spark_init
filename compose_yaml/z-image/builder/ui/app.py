"""Z-Image Gradio Web UI"""

import argparse
import atexit
import logging
import os
import time
from pathlib import Path

import gradio as gr

from generators import (
    generate_turbo,
    generate_base,
    generate_img2img,
    get_gen_info_for_tab,
    get_loading_status,
    get_worker_mgr,
    set_model_dir,
)
from pipeline_manager import (
    OUTPUT_DIR,
    RESOLUTION_CHOICES,
    SAMPLE_PROMPTS,
    scan_lora_files,
)
from config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zimage-ui")

MAX_CUSTOM_LORAS = 3
LORAS_DIR = Path(MODEL_DIR) / "loras"


# ---------------------------------------------------------------------------
# Kill handler
# ---------------------------------------------------------------------------
def _do_kill():
    mgr = get_worker_mgr()
    msg = mgr.kill()
    logger.info("Kill: %s", msg)
    return msg


# ---------------------------------------------------------------------------
# LoRA section
# ---------------------------------------------------------------------------
def create_lora_section():
    """Multi-LoRA loader. Returns gr.State with list of dicts."""
    slots_dd = []
    slots_str = []
    slots_del = []
    rows = []
    L = MAX_CUSTOM_LORAS
    initial_choices = scan_lora_files()

    with gr.Row():
        add_btn = gr.Button("+ Add LoRA", size="sm", variant="secondary")
        refresh_btn = gr.Button("Refresh", size="sm", variant="secondary", min_width=80)

    for i in range(L):
        with gr.Group(visible=False) as grp:
            with gr.Row():
                dd = gr.Dropdown(choices=initial_choices, label=f"LoRA {i+1}", scale=3, allow_custom_value=False)
                stren = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Strength", scale=1)
                d_btn = gr.Button("✕", size="sm", variant="stop", min_width=40, scale=0)
        slots_dd.append(dd)
        slots_str.append(stren)
        slots_del.append(d_btn)
        rows.append(grp)

    count_state = gr.State(0)
    lora_state = gr.State([])

    def _add_slot(n):
        n = min(n + 1, L)
        return [n] + [gr.Group(visible=(i < n)) for i in range(L)]

    def _sync_state(*vals):
        dds = vals[:L]
        strs_ = vals[L:]
        return [{"filename": str(d), "strength": float(s)} for d, s in zip(dds, strs_) if d]

    def _make_delete(slot_idx):
        def _delete(n, *vals):
            dds = list(vals[:L])
            strs_ = list(vals[L:])
            for j in range(slot_idx, n - 1):
                dds[j], strs_[j] = dds[j+1], strs_[j+1]
            last = n - 1
            dds[last], strs_[last] = None, 1.0
            n = max(n - 1, 0)
            vis = [gr.Group(visible=(i < n)) for i in range(L)]
            result = [{"filename": str(d), "strength": float(s)} for d, s in zip(dds, strs_) if d]
            return [n] + vis + dds + strs_ + [result]
        return _delete

    def _refresh():
        return [gr.Dropdown(choices=scan_lora_files()) for _ in range(L)]

    add_btn.click(fn=_add_slot, inputs=[count_state], outputs=[count_state] + rows)
    refresh_btn.click(fn=_refresh, inputs=[], outputs=slots_dd)

    all_inputs = slots_dd + slots_str
    all_outputs = [count_state] + rows + slots_dd + slots_str + [lora_state]
    for i in range(L):
        slots_del[i].click(fn=_make_delete(i), inputs=[count_state] + all_inputs, outputs=all_outputs)
    for comp in all_inputs:
        comp.change(fn=_sync_state, inputs=all_inputs, outputs=[lora_state])

    return lora_state


# ---------------------------------------------------------------------------
# Output column
# ---------------------------------------------------------------------------
def create_output_column(gen_type: str):
    image = gr.Image(label="Generated Image", type="filepath")
    info = gr.Textbox(
        label="Info", interactive=False,
        value=lambda: get_gen_info_for_tab(gen_type), every=2,
    )
    with gr.Row():
        kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm")
    kill_msg = gr.Textbox(label="", interactive=False, visible=False)
    status_md = gr.Markdown(value=get_loading_status, every=1)
    kill_btn.click(fn=_do_kill, outputs=[kill_msg])
    return image, info


# ---------------------------------------------------------------------------
# History tab helpers
# ---------------------------------------------------------------------------
def _list_outputs():
    out_dir = Path(OUTPUT_DIR)
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("zimage_*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
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
    for f in out_dir.glob("zimage_*"):
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

    with gr.Blocks(
        title="Z-Image Gradio UI",
        css=".memory-status { text-align: right; } .kill-btn { background: #dc2626 !important; }",
    ) as app:
        with gr.Row():
            gr.Markdown("# Z-Image Gradio UI")
            gr.Markdown(value=get_memory_status, every=3, elem_classes=["memory-status"])

        with gr.Tabs():
            # ==============================================================
            # Tab 1: Turbo (8-step fast)
            # ==============================================================
            with gr.Tab("Turbo"):
                gr.Markdown("*Fast 8-step generation (Z-Image-Turbo). No CFG, no negative prompt.*")
                with gr.Row():
                    with gr.Column(scale=1):
                        t1_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your image...")
                        with gr.Row():
                            for i, sp in enumerate(SAMPLE_PROMPTS[:3]):
                                gr.Button(f"Sample {i+1}", size="sm", min_width=60).click(
                                    fn=lambda s=sp: s, outputs=[t1_prompt])
                        t1_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="1024x1024", label="Resolution (WxH)", allow_custom_value=True)
                        with gr.Row():
                            t1_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                            t1_num = gr.Number(value=1, label="Num Images", precision=0, minimum=1, maximum=4)
                        with gr.Accordion("LoRA", open=False):
                            t1_lora_state = create_lora_section()
                        t1_generate = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        t1_image, t1_info = create_output_column("turbo")

                t1_generate.click(
                    fn=generate_turbo,
                    inputs=[t1_prompt, t1_resolution, t1_seed, t1_num, t1_lora_state],
                    outputs=[t1_image, t1_info],
                )

            # ==============================================================
            # Tab 2: Base (CFG, negative prompt)
            # ==============================================================
            with gr.Tab("Base"):
                gr.Markdown("*Z-Image Base model. Supports CFG guidance and negative prompts. 28-50 steps recommended.*")
                with gr.Row():
                    with gr.Column(scale=1):
                        t2_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your image...")
                        t2_neg = gr.Textbox(label="Negative Prompt", lines=2, placeholder="blurry, distorted, low quality")
                        with gr.Row():
                            for i, sp in enumerate(SAMPLE_PROMPTS[:3]):
                                gr.Button(f"Sample {i+1}", size="sm", min_width=60).click(
                                    fn=lambda s=sp: s, outputs=[t2_prompt])
                        t2_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="1024x1024", label="Resolution (WxH)", allow_custom_value=True)
                        with gr.Row():
                            t2_steps = gr.Slider(10, 100, value=28, step=1, label="Steps")
                            t2_cfg = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="Guidance Scale")
                        with gr.Row():
                            t2_cfg_norm = gr.Checkbox(label="CFG Normalization", value=False, info="False=stylism, True=realism")
                            t2_cfg_trunc = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="CFG Truncation")
                        with gr.Row():
                            t2_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                            t2_num = gr.Number(value=1, label="Num Images", precision=0, minimum=1, maximum=4)
                        with gr.Accordion("LoRA", open=False):
                            t2_lora_state = create_lora_section()
                        t2_generate = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        t2_image, t2_info = create_output_column("base")

                t2_generate.click(
                    fn=generate_base,
                    inputs=[t2_prompt, t2_neg, t2_resolution, t2_steps, t2_cfg,
                            t2_cfg_norm, t2_cfg_trunc, t2_seed, t2_num, t2_lora_state],
                    outputs=[t2_image, t2_info],
                )

            # ==============================================================
            # Tab 3: Image → Image
            # ==============================================================
            with gr.Tab("Img2Img"):
                gr.Markdown("*Transform an existing image with a text prompt.*")
                with gr.Row():
                    with gr.Column(scale=1):
                        t3_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the transformation...")
                        t3_neg = gr.Textbox(label="Negative Prompt", lines=2)
                        t3_image_in = gr.Image(label="Input Image", type="numpy")
                        t3_strength = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Strength",
                                                info="0=no change, 1=full regeneration")
                        t3_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="1024x1024", label="Resolution (WxH)", allow_custom_value=True)
                        with gr.Row():
                            t3_steps = gr.Slider(5, 100, value=20, step=1, label="Steps")
                            t3_cfg = gr.Slider(0.0, 10.0, value=0.0, step=0.5, label="Guidance Scale",
                                               info="0=Turbo mode, >1=Base mode with CFG")
                        t3_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        t3_use_base = gr.Checkbox(label="Use Base Model", value=False,
                                                  info="Use Base model for CFG support")
                        with gr.Accordion("LoRA", open=False):
                            t3_lora_state = create_lora_section()
                        t3_generate = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        t3_image_out, t3_info = create_output_column("img2img")

                t3_generate.click(
                    fn=generate_img2img,
                    inputs=[t3_prompt, t3_neg, t3_image_in, t3_strength,
                            t3_resolution, t3_steps, t3_cfg, t3_seed,
                            t3_use_base, t3_lora_state],
                    outputs=[t3_image_out, t3_info],
                )

            # ==============================================================
            # Tab 4: Settings
            # ==============================================================
            with gr.Tab("Settings"):
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
                    s_check_status = gr.Textbox(label="Model Status", interactive=False, lines=6)

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
                    s_lora_url = gr.Textbox(label="HuggingFace Repo ID or URL (.safetensors)")
                    s_lora_fname = gr.Textbox(label="Filename in Repo (for HF, e.g. model.safetensors)")
                    s_lora_save = gr.Textbox(label="Save As (optional)")
                    s_lora_dl = gr.Button("Download", variant="secondary", size="sm")
                    s_lora_status = gr.Textbox(label="Download Status", interactive=False)

                    def _download_lora(source, fname, save_as):
                        loras_dir = Path(MODEL_DIR) / "loras"
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

                    s_lora_dl.click(fn=_download_lora,
                                   inputs=[s_lora_url, s_lora_fname, s_lora_save],
                                   outputs=[s_lora_status])

                gr.Markdown("### Installed LoRAs")
                s_lora_list = gr.Dataframe(
                    headers=["Filename", "Size"],
                    value=lambda: [[f.name, f"{f.stat().st_size/1024/1024:.1f} MB"]
                                   for f in sorted((Path(MODEL_DIR)/"loras").glob("*.safetensors"))]
                        if (Path(MODEL_DIR)/"loras").exists() else [],
                    interactive=False, every=10,
                )

            # ==============================================================
            # Tab 5: History
            # ==============================================================
            with gr.Tab("History"):
                gr.Markdown("### Generation History")
                with gr.Row():
                    h_refresh = gr.Button("Refresh", size="sm")
                    h_delete = gr.Button("Delete Selected", size="sm", variant="stop")
                    h_delete_all = gr.Button("Delete All", size="sm", variant="stop")
                    h_clear_cache = gr.Button("Clear Cache", size="sm")

                h_gallery = gr.Gallery(label="Generated Images", value=_list_outputs,
                                       columns=4, height=400, object_fit="contain",
                                       every=10)
                h_selected = gr.Textbox(label="Selected File", interactive=False, visible=False)
                h_file_info = gr.Textbox(label="File Info", interactive=False, lines=8)
                h_cache_msg = gr.Textbox(label="", interactive=False, visible=False)

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

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Z-Image Gradio UI")
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
