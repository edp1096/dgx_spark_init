"""Settings tab for ZIT UI."""

from pathlib import Path

import gradio as gr

from generators import set_model_dir
from helpers import lora_list
from i18n import LANGUAGES
from zit_config import MODEL_DIR


def build_settings_tab():
    """Build Settings tab UI and wire events."""
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
            value=lambda: lora_list(),
            interactive=False, every=10,
            max_height=200,
        )
        with gr.Row():
            s_lora_selected = gr.Textbox(label="Selected", interactive=False, scale=3)
            s_lora_del_btn = gr.Button("Delete Selected", variant="stop", size="sm", scale=1)
        s_lora_del_status = gr.Textbox(label="", interactive=False, visible=False)

        def _on_lora_select(evt: gr.SelectData):
            loras = lora_list()
            idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if isinstance(idx, int) and 0 <= idx < len(loras):
                return loras[idx][0]
            return ""

        def _delete_lora(filename):
            try:
                if not filename or not filename.strip():
                    return lora_list(), "", "No file selected."
                filename = filename.strip()
                from zit_config import LORAS_DIR
                lora_path = Path(MODEL_DIR) / LORAS_DIR / filename
                if not lora_path.exists():
                    return lora_list(), "", f"Not found: {filename}"
                lora_path.unlink()
                return lora_list(), "", f"Deleted: {filename}"
            except Exception as e:
                return lora_list(), "", f"Error: {e}"

        s_lora_table.select(fn=_on_lora_select, outputs=[s_lora_selected])
        s_lora_del_btn.click(
            fn=_delete_lora,
            inputs=[s_lora_selected],
            outputs=[s_lora_table, s_lora_selected, s_lora_del_status],
        )
