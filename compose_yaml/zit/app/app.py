"""ZIT Gradio Web UI — Z-Image-Turbo image generation."""

import argparse
import atexit
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "ui"))

import gradio as gr

from generators import (
    get_gen_info_for_tab,
    get_loading_status,
    get_worker_mgr,
)
from helpers import get_memory_status
from i18n import get_i18n_js
from tab_generate import build_generate_tab
from tab_inpaint import build_inpaint_tab
from tab_train import build_train_tab
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
#history-page-info { text-align: right !important; }
#history-page-info .prose { text-align: right !important; }
#history-info-col { display: flex; flex-direction: column; height: 100%; }
#history-file-info { flex: 1; display: flex; flex-direction: column; height: 100% !important; }
#history-file-info label { flex: 1; display: flex; flex-direction: column; height: 100%; }
#history-file-info textarea { flex: 1; height: 100% !important; }
/* History: remove bottom dead space — target gallery container only */
#history-gallery .grid-wrap.fixed-height {
  max-height: calc(100vh - 300px) !important;
  overflow-y: auto !important;
}
#history-gallery .preview img {
  max-height: calc(100vh - 300px) !important;
  object-fit: contain;
}
#history-file-info textarea { height: calc(100vh - 400px) !important; min-height: 100px; }
@media (max-width: 768px) {
  #history-gallery .thumbnails { grid-template-columns: repeat(2, 1fr) !important; }
}
#presets-section .gallery { transition: max-height 0.3s ease; }
#presets-toggle-row { margin-bottom: 4px; }
#presets-toggle-row button { min-width: 80px !important; }
/* Settings sidebar TOC active highlight */
.toc-active { background: var(--button-primary-background-fill) !important;
  color: var(--button-primary-text-color) !important;
  border-color: var(--button-primary-border-color) !important; }
"""


# ---------------------------------------------------------------------------
# UI Builder
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ZIT Gradio", analytics_enabled=False) as app:
        with gr.Row():
            gr.Markdown("# ZIT Gradio")
            memory_md = gr.Markdown(elem_classes=["memory-status"])

        # Settings sidebar (gr.Sidebar stays fixed on screen)
        settings_sidebar = gr.Sidebar(
            open=False, visible=False, width=180,
            elem_id="settings-sidebar",
        )

        with gr.Tabs() as tabs:
            with gr.Tab("Generate", id="generate") as gen_tab:
                gen = build_generate_tab()

            with gr.Tab("Inpaint", id="inpaint") as ip_tab:
                ip = build_inpaint_tab()

            with gr.Tab("History", id="history") as h_tab:
                build_history_tab(h_tab)

            with gr.Tab("Train", id="train") as tr_tab:
                tr = build_train_tab(tr_tab)

            with gr.Tab("Settings", id="settings") as settings_tab:
                build_settings_tab(settings_sidebar)

        # ---------------------------------------------------------------
        # Tab select: refresh LoRA list when switching to Generate/Inpaint
        # ---------------------------------------------------------------
        from helpers import lora_choices
        from zit_config import MAX_LORA_STACK

        def _refresh_lora_dropdowns():
            choices = lora_choices()
            return [gr.update(choices=choices)] * MAX_LORA_STACK

        gen_tab.select(fn=_refresh_lora_dropdowns, outputs=gen["lora_dropdowns"])
        ip_tab.select(fn=_refresh_lora_dropdowns, outputs=ip["lora_dropdowns"])

        # ---------------------------------------------------------------
        # Settings sidebar: show only when Settings tab is active
        # ---------------------------------------------------------------
        def _show_sidebar():
            return gr.update(open=False, visible=True)

        _open_sidebar_js = """
        () => {
            if (window.innerWidth > 768) {
                requestAnimationFrame(() => {
                    const sb = document.querySelector('#settings-sidebar');
                    if (sb && !sb.classList.contains('open')) {
                        const btn = sb.querySelector('button.toggle-button');
                        if (btn) btn.click();
                    }
                });
            }
        }
        """

        def _hide_sidebar():
            return gr.update(open=False, visible=False)

        settings_tab.select(fn=_show_sidebar, outputs=[settings_sidebar],
                            js=_open_sidebar_js)
        gen_tab.select(fn=_hide_sidebar, outputs=[settings_sidebar])
        ip_tab.select(fn=_hide_sidebar, outputs=[settings_sidebar])
        tr_tab.select(fn=_hide_sidebar, outputs=[settings_sidebar])
        h_tab.select(fn=_hide_sidebar, outputs=[settings_sidebar])

        # ---------------------------------------------------------------
        # NOTE: No app.load() — it triggers full Gradio re-render on
        # page load, destroying ImageEditor's PixiJS canvas.
        # Refresh recovery is not supported; use History tab instead.
        # ---------------------------------------------------------------

        # ---------------------------------------------------------------
        # Polling via gr.Timer — MUST be outside Tab context.
        # Timer inside a Tab causes Gradio to re-render the entire tab
        # on each tick, resetting ImageEditor internal state (brush size).
        # ---------------------------------------------------------------
        _timer_3s = gr.Timer(3)
        _timer_2s = gr.Timer(2)
        _timer_1s = gr.Timer(1)
        _timer_3s.tick(fn=get_memory_status, outputs=[memory_md])
        _timer_2s.tick(fn=lambda: get_gen_info_for_tab("generate"), outputs=[gen["info"]])
        _timer_2s.tick(fn=lambda: get_gen_info_for_tab("inpaint"), outputs=[ip["info"]])
        _timer_1s.tick(fn=get_loading_status, outputs=[gen["loading_md"]])
        _timer_1s.tick(fn=get_loading_status, outputs=[ip["loading_md"]])

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
