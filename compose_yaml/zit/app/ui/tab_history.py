"""History tab for ZIT UI."""

import logging

import gradio as gr

from helpers import (
    list_outputs, get_file_info, delete_selected,
    download_all, delete_all, clear_cache, extract_gallery_path,
)

logger = logging.getLogger("zit-ui")


def build_history_tab(tab_ref):
    """Build History tab UI and wire events."""
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
                label="Generated Images", value=list_outputs,
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
        path = extract_gallery_path(evt)
        idx = evt.index if isinstance(evt.index, int) else 0
        logger.info("Gallery select index=%r path=%r", evt.index, path)
        if path:
            return path, idx, get_file_info(path)
        return "", idx, ""

    def _on_history_tab():
        """Auto-select first image when entering History tab."""
        outputs = list_outputs()
        if outputs:
            return outputs[0], 0, get_file_info(outputs[0])
        return "", 0, ""

    h_gallery.select(fn=_on_gallery_select, outputs=[h_selected, h_sel_idx, h_file_info])
    tab_ref.select(fn=_on_history_tab, outputs=[h_selected, h_sel_idx, h_file_info])
    h_refresh.click(fn=list_outputs, outputs=[h_gallery])
    h_delete.click(
        fn=delete_selected, inputs=[h_selected, h_sel_idx],
        outputs=[h_gallery, h_selected, h_sel_idx, h_file_info],
    )
    h_download_all.click(fn=download_all, outputs=[h_download_file])
    h_delete_all.click(fn=delete_all, outputs=[h_gallery])
    h_clear_cache.click(fn=clear_cache, outputs=[h_cache_msg])
