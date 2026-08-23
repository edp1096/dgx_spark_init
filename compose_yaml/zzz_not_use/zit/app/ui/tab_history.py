"""History tab for ZIT UI."""

import logging

import gradio as gr

from helpers import (
    list_outputs_paged, get_file_info, delete_selected,
    download_all, delete_all, clear_cache, extract_gallery_path,
)

logger = logging.getLogger("zit-ui")


def build_history_tab(tab_ref):
    """Build History tab UI and wire events."""
    h_page = gr.State(0)

    with gr.Row():
        gr.Markdown("### Generation History")
        h_page_info = gr.Markdown("1 / 1 (0)", elem_id="history-page-info")
    with gr.Row():
        h_first = gr.Button("<<", size="sm", min_width=36)
        h_prev3 = gr.Button("-3", size="sm", min_width=36)
        h_prev = gr.Button("<", size="sm", min_width=36)
        h_next = gr.Button(">", size="sm", min_width=36)
        h_next3 = gr.Button("+3", size="sm", min_width=36)
        h_last = gr.Button(">>", size="sm", min_width=36)
        h_refresh = gr.Button("Refresh", size="sm")
        h_download_all = gr.Button("Download All", size="sm", variant="secondary")
        h_delete = gr.Button("Delete Selected", size="sm", variant="stop")
        h_delete_all = gr.Button("Delete All", size="sm", variant="stop")
        h_clear_cache = gr.Button("Clear Cache", size="sm")

    with gr.Row(equal_height=True, elem_id="history-row"):
        with gr.Column(scale=3):
            h_gallery = gr.Gallery(
                label="Generated Images", value=[],
                columns=4, height="auto", object_fit="contain",
                elem_id="history-gallery", preview=False,
                selected_index=None, buttons=["download", "fullscreen"],
            )
        with gr.Column(scale=1, elem_id="history-info-col"):
            h_selected = gr.Textbox(label="Selected File", interactive=False, visible=False)
            h_sel_idx = gr.State(0)
            h_file_info = gr.Textbox(label="File Info", interactive=False, lines=12,
                                     elem_id="history-file-info")
            h_download_file = gr.File(label="Download", visible=False, interactive=False)
            h_cache_msg = gr.Textbox(label="", interactive=False, visible=False)

    def _page_info_text(page, total_pages, total):
        return f"**{page + 1}** / {total_pages} ({total})"

    def _load_page(page, select_idx=0):
        files, page, total_pages, total = list_outputs_paged(page)
        info = _page_info_text(page, total_pages, total)
        if not files:
            return gr.Gallery(value=[], selected_index=None), page, info, "", 0, ""
        idx = min(select_idx, len(files) - 1)
        selected = files[idx]
        return gr.Gallery(value=files, selected_index=idx), page, info, selected, idx, get_file_info(selected)

    def _go_first(_):
        return _load_page(0)

    def _go_prev(page):
        return _load_page(max(0, page - 1))

    def _go_prev3(page):
        return _load_page(max(0, page - 3))

    def _go_next(page):
        return _load_page(page + 1)

    def _go_next3(page):
        return _load_page(page + 3)

    def _go_last(_):
        _, _, total_pages, _ = list_outputs_paged(0)
        return _load_page(total_pages - 1)

    def _refresh(page, sel_idx=0):
        return _load_page(page, select_idx=sel_idx)

    _page_outputs = [h_gallery, h_page, h_page_info, h_selected, h_sel_idx, h_file_info]

    h_first.click(fn=_go_first, inputs=[h_page], outputs=_page_outputs)
    h_prev3.click(fn=_go_prev3, inputs=[h_page], outputs=_page_outputs)
    h_prev.click(fn=_go_prev, inputs=[h_page], outputs=_page_outputs)
    h_next.click(fn=_go_next, inputs=[h_page], outputs=_page_outputs)
    h_next3.click(fn=_go_next3, inputs=[h_page], outputs=_page_outputs)
    h_last.click(fn=_go_last, inputs=[h_page], outputs=_page_outputs)
    h_refresh.click(fn=_refresh, inputs=[h_page, h_sel_idx], outputs=_page_outputs)

    def _on_gallery_select(evt: gr.SelectData):
        path = extract_gallery_path(evt)
        idx = evt.index if isinstance(evt.index, int) else 0
        logger.info("Gallery select index=%r path=%r", evt.index, path)
        if path:
            return path, idx, get_file_info(path)
        return "", idx, ""

    def _on_history_tab(page):
        files, pg, total_pages, total = list_outputs_paged(page)
        info = _page_info_text(pg, total_pages, total)
        return gr.Gallery(value=files, selected_index=None), pg, info, "", 0, ""

    h_gallery.select(fn=_on_gallery_select, outputs=[h_selected, h_sel_idx, h_file_info])
    tab_ref.select(fn=_on_history_tab, inputs=[h_page], outputs=_page_outputs)

    def _delete_and_reload(file_path, sel_idx, page):
        from pathlib import Path
        if not file_path:
            return _load_page(page)
        p = Path(file_path)
        if p.exists():
            p.unlink()
        json_p = p.with_suffix(".json")
        if json_p.exists():
            json_p.unlink()
        return _load_page(page, select_idx=sel_idx)

    h_delete.click(
        fn=_delete_and_reload, inputs=[h_selected, h_sel_idx, h_page],
        outputs=_page_outputs,
    )

    def _delete_all_and_reload():
        delete_all()
        return _load_page(0)

    h_delete_all.click(fn=_delete_all_and_reload, outputs=_page_outputs)
    h_download_all.click(fn=download_all, outputs=[h_download_file])
    h_clear_cache.click(fn=clear_cache, outputs=[h_cache_msg])

