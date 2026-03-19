"""ZIT Gradio Web UI — Z-Image-Turbo image generation."""

import argparse
import atexit
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "ui"))

import gradio as gr

from generators import (
    get_gen_ui_params,
    get_worker_mgr,
    wait_for_gen_completion,
)
from helpers import get_memory_status
from i18n import get_i18n_js
from tab_generate import build_generate_tab
from tab_inpaint import build_inpaint_tab
from tab_train import build_train_tab, get_restore_train_params
from tab_settings import build_settings_tab
from tab_history import build_history_tab

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zit-ui")


# ---------------------------------------------------------------------------
# Custom CSS (references elem_ids from all tabs)
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
.memory-status { text-align: right; }
#gen-gallery .grid-container,
#history-gallery .grid-container,
#presets-gallery .grid-container,
#dataset-gallery .grid-container {
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)) !important;
}
#gen-gallery .thumbnails button,
#history-gallery .thumbnails button,
#presets-gallery .thumbnails button,
#dataset-gallery .thumbnails button {
  max-height: 200px;
  max-width: 200px;
}
#gen-gallery .thumbnails button img,
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
/* Settings TOC active highlight */
.toc-active { background: var(--button-primary-background-fill) !important;
  color: var(--button-primary-text-color) !important;
  border-color: var(--button-primary-border-color) !important; }
/* Settings TOC sticky positioning */
#settings-row { align-items: flex-start !important; }
#settings-toc { position: sticky !important; top: 0; align-self: flex-start; max-height: 100vh; overflow-y: auto !important; }
@media (max-width: 768px) { #settings-toc { position: static; max-height: none; } }
"""


# ---------------------------------------------------------------------------
# UI Builder
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ZIT Gradio", analytics_enabled=False) as app:
        with gr.Row():
            gr.Markdown("# ZIT Gradio")
            gr.Markdown(value=get_memory_status, every=3, elem_classes=["memory-status"])

        with gr.Tabs() as tabs:
            with gr.Tab("Generate", id="generate") as gen_tab:
                gen = build_generate_tab()

            with gr.Tab("Inpaint", id="inpaint") as ip_tab:
                ip = build_inpaint_tab()

            with gr.Tab("Train", id="train") as tr_tab:
                tr = build_train_tab(tr_tab)

            with gr.Tab("Settings", id="settings"):
                build_settings_tab()

            with gr.Tab("History", id="history") as h_tab:
                build_history_tab(h_tab)

        # ---------------------------------------------------------------
        # Tab select: refresh LoRA list when switching to Generate/Inpaint
        # ---------------------------------------------------------------
        from helpers import lora_choices
        from zit_config import MAX_LORA_STACK

        def _refresh_lora_dropdowns():
            choices = lora_choices()
            return [gr.Dropdown(choices=choices)] * MAX_LORA_STACK

        gen_tab.select(fn=_refresh_lora_dropdowns, outputs=gen["lora_dropdowns"])
        ip_tab.select(fn=_refresh_lora_dropdowns, outputs=ip["lora_dropdowns"])

        # ---------------------------------------------------------------
        # Page load: restore params if generation/training is in progress
        # ---------------------------------------------------------------
        def _restore_gen_params():
            """Restore Generate tab params on refresh during generation."""
            skip = tuple([gr.update()] * 14)
            is_active, gen_type, p = get_gen_ui_params()
            if not is_active or not p or p.get("tab") != "generate":
                return skip
            return (
                p["prompt"], p["neg"], p["resolution"],
                p["seed"], p["num_images"],
                p["steps"], p["time_shift"],
                p["cfg"], p["cfg_norm"], p["cfg_trunc"],
                p["max_seq"], p["use_fp8"], p["attn"],
                p["lora_enable"],
            )

        def _restore_ip_params():
            """Restore Inpaint tab params on refresh during generation."""
            skip = tuple([gr.update()] * 12)
            is_active, gen_type, p = get_gen_ui_params()
            if not is_active or not p or p.get("tab") != "inpaint":
                return skip
            return (
                p["prompt"], p["neg"], p["resolution"],
                p["seed"],
                p["steps"], p["time_shift"],
                p["control_scale"], p["guidance"],
                p["cfg_trunc"], p["max_seq"],
                p["use_controlnet"],
                p["lora_enable"],
            )

        def _recover_gallery():
            """Block until ongoing generation completes, then update gallery."""
            is_active, _, _ = get_gen_ui_params()
            if not is_active:
                return gr.update(), gr.update()

            result = wait_for_gen_completion(timeout=600)
            if not result:
                return gr.update(), gr.update()

            paths = result["paths"]
            gen_type = result["gen_type"]
            if gen_type in ("zit_t2i", "controlnet"):
                return gr.Gallery(value=paths, selected_index=0), gr.update()
            elif gen_type in ("inpaint", "outpaint"):
                return gr.update(), paths[0] if paths else None
            return gr.update(), gr.update()

        app.load(
            fn=_restore_gen_params,
            outputs=[gen["prompt"], gen["neg"], gen["resolution"],
                     gen["seed"], gen["num_images"],
                     gen["steps"], gen["time_shift"],
                     gen["cfg"], gen["cfg_norm"], gen["cfg_trunc"],
                     gen["max_seq"], gen["use_fp8"], gen["attn"],
                     gen["lora_enable"]],
        )
        app.load(
            fn=_restore_ip_params,
            outputs=[ip["prompt"], ip["neg"], ip["resolution"],
                     ip["seed"],
                     ip["steps"], ip["time_shift"],
                     ip["control_scale"], ip["guidance"],
                     ip["cfg_trunc"], ip["max_seq"],
                     ip["use_controlnet"],
                     ip["lora_enable"]],
        )
        app.load(
            fn=_recover_gallery,
            outputs=[gen["gallery"], ip["result"]],
        )
        app.load(
            fn=get_restore_train_params,
            outputs=[tr["dataset"], tr["name"], tr["steps"], tr["rank"],
                     tr["lr"], tr["lora_alpha"], tr["resolution"],
                     tr["batch"], tr["grad_accum"], tr["save_every"], tr["targets"]],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
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
        css=_CUSTOM_CSS,
        js=get_i18n_js(),
    )


if __name__ == "__main__":
    main()
