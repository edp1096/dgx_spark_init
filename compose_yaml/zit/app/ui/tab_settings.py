"""Settings tab for ZIT UI."""

import logging
import os
import re
from pathlib import Path

import gradio as gr

from generators import set_model_dir
from helpers import (
    lora_list, lora_list_with_info,
    get_lora_info, update_lora_info, load_lora_metadata, save_lora_metadata,
)
from i18n import LANGUAGES
from zit_config import MODEL_DIR, CIVITAI_API_BASE

logger = logging.getLogger("zit-ui")

# Menu item definitions: (id, label)
_MENU_ITEMS = [
    ("lang", "Language"),
    ("model", "Model Settings"),
    ("lora_dl", "LoRA Download"),
    ("lora_up", "LoRA Upload"),
    ("lora_list", "Installed LoRAs"),
]


def build_settings_tab():
    """Build Settings tab UI and wire events."""

    # ==================================================================
    # Layout: left TOC + right content (all sections visible)
    # ==================================================================
    with gr.Row():
        # --- Left TOC column ---
        with gr.Column(scale=0, min_width=160, elem_id="settings-toc"):
            gr.Markdown("### Settings")
            nav_buttons = {}
            for item_id, label in _MENU_ITEMS:
                nav_buttons[item_id] = gr.Button(
                    label, variant="secondary", size="sm",
                    elem_id=f"nav-{item_id}",
                )

        # --- Right content column ---
        with gr.Column(scale=4):

            # ============================================================
            # Section: Language
            # ============================================================
            with gr.Column(elem_id="sec-lang") as sec_lang:
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
                            const map = {""" + ", ".join(
                                f'"{v}": "{k}"' for k, v in LANGUAGES.items()
                            ) + """};
                            const code = map[lang] || 'en';
                            if (window._zit_setLang) window._zit_setLang(code);
                        }""",
                    )

            # ============================================================
            # Section: Model Settings
            # ============================================================
            with gr.Column(elem_id="sec-model") as sec_model:
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

            # ============================================================
            # Section: LoRA Download
            # ============================================================
            with gr.Column(elem_id="sec-lora-dl") as sec_lora_dl:
                gr.Markdown("### LoRA Download")
                with gr.Group():
                    s_dl_url = gr.Textbox(
                        label="URL / HuggingFace Repo ID / CivitAI URL",
                        placeholder="https://civitai.com/models/... or user/repo or https://...",
                    )
                    s_dl_fname = gr.Textbox(
                        label="Filename in Repo",
                        placeholder="e.g. model.safetensors (HuggingFace only)",
                    )
                    s_dl_save = gr.Textbox(label="Save As (optional)")
                    s_dl_trigger = gr.Textbox(
                        label="Trigger Words",
                        placeholder="e.g. lya, lee young-ae (CivitAI: auto-filled)",
                    )
                    s_dl_rec_scale = gr.Slider(
                        0.0, 3.0, value=1.0, step=0.05, label="Recommend Scale",
                    )
                    s_dl_civitai_key = gr.Textbox(
                        label="CivitAI API Key (CivitAI only)",
                        type="password",
                        value=os.environ.get("CIVITAI_API_KEY", ""),
                        placeholder="Required for CivitAI downloads",
                    )
                    s_dl_btn = gr.Button("Download", variant="primary", size="sm")
                    s_dl_status = gr.Textbox(label="Status", interactive=False, lines=3)

                    def _is_civitai(source: str) -> bool:
                        return "civitai.com" in source or source.strip().isdigit()

                    def _download_lora(source, fname, save_as, trigger_words, rec_scale, api_key):
                        from zit_config import LORAS_DIR
                        import urllib.request

                        source = source.strip()
                        if not source:
                            return "Error: URL or Repo ID required", ""

                        loras_dir = Path(MODEL_DIR) / LORAS_DIR
                        loras_dir.mkdir(parents=True, exist_ok=True)

                        try:
                            if _is_civitai(source):
                                return _download_civitai(
                                    source, api_key, save_as, trigger_words, rec_scale, loras_dir,
                                )

                            # HuggingFace repo or direct URL
                            if source.startswith("http"):
                                out_name = save_as.strip() if save_as.strip() else source.split("/")[-1]
                                dest = loras_dir / out_name
                                urllib.request.urlretrieve(source, str(dest))
                            else:
                                from huggingface_hub import hf_hub_download
                                filename = fname.strip() if fname.strip() else "model.safetensors"
                                out_name = save_as.strip() if save_as.strip() else filename
                                hf_hub_download(source, filename, local_dir=str(loras_dir))
                                if filename != out_name:
                                    src = loras_dir / filename
                                    if src.exists():
                                        src.rename(loras_dir / out_name)
                                dest = loras_dir / out_name

                            _auto_populate_metadata(dest.name, dest)
                            meta_updates = {}
                            if trigger_words.strip():
                                meta_updates["trigger_words"] = trigger_words.strip()
                            if rec_scale != 1.0:
                                meta_updates["recommend_scale"] = float(rec_scale)
                            if meta_updates:
                                update_lora_info(dest.name, meta_updates)

                            return f"Downloaded: {dest.name}", ""

                        except Exception as e:
                            logger.error("LoRA download error: %s", e)
                            return f"Error: {e}", ""

                    def _download_civitai(civitai_url, api_key, save_as, trigger_words,
                                          rec_scale, loras_dir):
                        import urllib.request
                        import json as _json

                        if not api_key.strip():
                            return "Error: CivitAI API key required", ""

                        version_id = _parse_civitai_url(civitai_url.strip())
                        if not version_id:
                            return "Error: Could not parse CivitAI URL/ID", ""

                        api_url = f"{CIVITAI_API_BASE}/model-versions/{version_id}"
                        req = urllib.request.Request(api_url)
                        req.add_header("Authorization", f"Bearer {api_key.strip()}")
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            api_data = _json.loads(resp.read())

                        # Find primary safetensors file
                        dl_url = dl_name = None
                        for f in api_data.get("files", []):
                            if f.get("name", "").endswith(".safetensors"):
                                dl_url = f.get("downloadUrl")
                                dl_name = f.get("name")
                                break
                        if not dl_url:
                            return "Error: No safetensors file found in this model version", ""

                        out_name = save_as.strip() if save_as.strip() else dl_name
                        if not out_name.endswith(".safetensors"):
                            out_name += ".safetensors"

                        # Trigger words from API
                        api_triggers = [tw.strip() for tw in api_data.get("trainedWords", []) if tw.strip()]
                        api_trigger_str = ", ".join(api_triggers)

                        # Download
                        dest = loras_dir / out_name
                        dl_req = urllib.request.Request(dl_url)
                        dl_req.add_header("Authorization", f"Bearer {api_key.strip()}")
                        with urllib.request.urlopen(dl_req, timeout=600) as resp:
                            with open(dest, "wb") as f:
                                while True:
                                    chunk = resp.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    f.write(chunk)

                        _auto_populate_metadata(out_name, dest)
                        final_triggers = trigger_words.strip() if trigger_words.strip() else api_trigger_str
                        model_name = api_data.get("model", {}).get("name", "")
                        meta_updates = {
                            "trigger_words": final_triggers,
                            "description": model_name,
                            "source_url": civitai_url.strip(),
                        }
                        if rec_scale != 1.0:
                            meta_updates["recommend_scale"] = float(rec_scale)
                        update_lora_info(out_name, meta_updates)

                        size_mb = dest.stat().st_size / 1024 / 1024
                        status = f"Downloaded: {out_name} ({size_mb:.1f} MB)"
                        if api_trigger_str:
                            status += f"\nAPI trigger words: {api_trigger_str}"
                        return status, final_triggers

                    s_dl_btn.click(
                        fn=_download_lora,
                        inputs=[s_dl_url, s_dl_fname, s_dl_save, s_dl_trigger,
                                s_dl_rec_scale, s_dl_civitai_key],
                        outputs=[s_dl_status, s_dl_trigger],
                    )

            # ============================================================
            # Section: LoRA Upload
            # ============================================================
            with gr.Column(elem_id="sec-lora-up") as sec_lora_up:
                gr.Markdown("### LoRA Upload")
                with gr.Group():
                    s_upload_file = gr.File(
                        label="Upload .safetensors file",
                        file_types=[".safetensors"],
                        file_count="single",
                    )
                    s_upload_status = gr.Textbox(label="Status", interactive=False)

                    def _upload_lora(file):
                        if file is None:
                            return None, "No file selected"
                        import shutil
                        from zit_config import LORAS_DIR

                        src = Path(file.name if hasattr(file, "name") else file)
                        if not src.name.endswith(".safetensors"):
                            return None, "Error: only .safetensors files are supported"

                        loras_dir = Path(MODEL_DIR) / LORAS_DIR
                        loras_dir.mkdir(parents=True, exist_ok=True)
                        dest = loras_dir / src.name

                        if dest.exists():
                            return None, f"Error: {src.name} already exists. Delete existing file first or rename."

                        shutil.copy2(str(src), str(dest))
                        _auto_populate_metadata(dest.name, dest)

                        size_mb = dest.stat().st_size / 1024 / 1024
                        return None, f"Uploaded: {dest.name} ({size_mb:.1f} MB)"

            # ============================================================
            # Section: Installed LoRAs
            # ============================================================
            with gr.Column(elem_id="sec-lora-list") as sec_lora_list:
                gr.Markdown("### Installed LoRAs")
                with gr.Row():
                    with gr.Column(scale=1):
                        s_lora_table = gr.Dataframe(
                            headers=["Filename", "Size", "Trigger Words", "Source"],
                            value=lambda: lora_list_with_info(),
                            interactive=False, every=10,
                            max_height=250,
                        )
                        with gr.Row():
                            s_lora_selected = gr.Textbox(label="Selected", interactive=False, scale=3)
                            s_lora_del_btn = gr.Button("Delete", variant="stop", size="sm", scale=1)
                            s_lora_refresh_btn = gr.Button("Refresh", variant="secondary", size="sm", scale=1)
                        s_lora_del_status = gr.Textbox(label="", interactive=False, visible=False)

                    with gr.Column(scale=1):
                        gr.Markdown("#### LoRA Detail")
                        s_detail_name = gr.Textbox(label="Filename", interactive=False)
                        s_detail_trigger = gr.Textbox(label="Trigger Words")
                        s_detail_desc = gr.Textbox(label="Description", lines=2)
                        s_detail_dataset = gr.Textbox(label="Dataset")
                        s_detail_source = gr.Textbox(label="Source URL", interactive=False)
                        with gr.Row():
                            s_detail_rank = gr.Textbox(label="Rank", interactive=False, scale=1)
                            s_detail_alpha = gr.Textbox(label="Alpha", interactive=False, scale=1)
                        s_detail_rec_scale = gr.Slider(
                            0.0, 3.0, value=1.0, step=0.05,
                            label="Recommend Scale",
                            info="LoRA 선택 시 자동 적용되는 기본 강도",
                        )
                        s_detail_notes = gr.Textbox(label="Notes", lines=3)
                        s_detail_save = gr.Button("Save Metadata", variant="primary", size="sm")
                        s_detail_status = gr.Textbox(label="", interactive=False, show_label=False)

    # ==================================================================
    # Navigation: TOC scroll-into-view via JS
    # ==================================================================
    _SEC_ELEM_IDS = {
        "lang": "sec-lang",
        "model": "sec-model",
        "lora_dl": "sec-lora-dl",
        "lora_up": "sec-lora-up",
        "lora_list": "sec-lora-list",
    }

    for item_id, _ in _MENU_ITEMS:
        elem_id = _SEC_ELEM_IDS[item_id]
        nav_buttons[item_id].click(
            fn=None,
            js=f"""() => {{
                const el = document.getElementById('{elem_id}');
                if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}""",
        )

    # ==================================================================
    # TOC: IntersectionObserver — highlight active section in sidebar
    # ==================================================================
    _toc_nav_ids = [f"nav-{item_id}" for item_id, _ in _MENU_ITEMS]
    _toc_sec_ids = [_SEC_ELEM_IDS[item_id] for item_id, _ in _MENU_ITEMS]
    _toc_pairs = list(zip(_toc_sec_ids, _toc_nav_ids))

    _observer_js = """
    <script>
    (function() {
      const pairs = """ + str([[s, n] for s, n in _toc_pairs]) + """;
      let current = pairs[0][1];

      function highlight(navId) {
        if (current === navId) return;
        pairs.forEach(([_, nid]) => {
          const btn = document.getElementById(nid);
          if (btn) btn.classList.toggle('toc-active', nid === navId);
        });
        current = navId;
      }

      // initial highlight
      highlight(pairs[0][1]);

      const observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const pair = pairs.find(([sid]) => sid === entry.target.id);
            if (pair) highlight(pair[1]);
          }
        }
      }, { rootMargin: '-10% 0px -80% 0px', threshold: 0 });

      // Observe after DOM is ready
      function attach() {
        let found = 0;
        pairs.forEach(([sid]) => {
          const el = document.getElementById(sid);
          if (el) { observer.observe(el); found++; }
        });
        if (found < pairs.length) setTimeout(attach, 500);
      }
      attach();
    })();
    </script>
    """
    gr.HTML(_observer_js, visible=False)

    # ==================================================================
    # Events: Installed LoRAs
    # ==================================================================
    def _on_lora_select(evt: gr.SelectData):
        loras = lora_list_with_info()
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if not isinstance(idx, int) or idx < 0 or idx >= len(loras):
            return ("",) * 8 + (1.0, "")
        filename = loras[idx][0]
        info = get_lora_info(filename)
        try:
            rec_scale = float(info.get("recommend_scale", 1.0))
        except (ValueError, TypeError):
            rec_scale = 1.0
        return (
            filename,
            filename,
            info.get("trigger_words", ""),
            info.get("description", ""),
            info.get("dataset", ""),
            info.get("source_url", ""),
            str(info.get("rank", "")),
            str(info.get("alpha", "")),
            rec_scale,
            info.get("notes", ""),
        )

    s_lora_table.select(
        fn=_on_lora_select,
        outputs=[s_lora_selected, s_detail_name, s_detail_trigger, s_detail_desc,
                 s_detail_dataset, s_detail_source, s_detail_rank, s_detail_alpha,
                 s_detail_rec_scale, s_detail_notes],
    )

    def _save_metadata(filename, trigger, desc, dataset, rec_scale, notes):
        if not filename:
            return "No LoRA selected"
        update_lora_info(filename, {
            "trigger_words": trigger.strip(),
            "description": desc.strip(),
            "dataset": dataset.strip(),
            "recommend_scale": float(rec_scale),
            "notes": notes.strip(),
        })
        return f"Saved metadata for {filename}"

    s_detail_save.click(
        fn=_save_metadata,
        inputs=[s_detail_name, s_detail_trigger, s_detail_desc,
                s_detail_dataset, s_detail_rec_scale, s_detail_notes],
        outputs=[s_detail_status],
    )

    def _delete_lora(filename):
        try:
            if not filename or not filename.strip():
                return lora_list_with_info(), "", "No file selected."
            filename = filename.strip()
            from zit_config import LORAS_DIR
            lora_path = Path(MODEL_DIR) / LORAS_DIR / filename
            if not lora_path.exists():
                return lora_list_with_info(), "", f"Not found: {filename}"
            lora_path.unlink()
            # Remove from metadata
            meta = load_lora_metadata()
            meta.pop(filename, None)
            save_lora_metadata(meta)
            return lora_list_with_info(), "", f"Deleted: {filename}"
        except Exception as e:
            return lora_list_with_info(), "", f"Error: {e}"

    s_lora_del_btn.click(
        fn=_delete_lora,
        inputs=[s_lora_selected],
        outputs=[s_lora_table, s_lora_selected, s_lora_del_status],
    )

    s_lora_refresh_btn.click(
        fn=lambda: lora_list_with_info(),
        outputs=[s_lora_table],
    )

    # Upload event — refresh table after upload
    def _upload_and_refresh(file):
        result_file, status = _upload_lora(file)
        return result_file, status, lora_list_with_info()

    s_upload_file.change(
        fn=_upload_and_refresh,
        inputs=[s_upload_file],
        outputs=[s_upload_file, s_upload_status, s_lora_table],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_civitai_url(url_or_id: str) -> str | None:
    """Parse CivitAI URL/ID to model version ID."""
    url_or_id = url_or_id.strip()
    if url_or_id.isdigit():
        return url_or_id

    # https://civitai.com/api/download/models/67890
    m = re.search(r'civitai\.com/api/download/models/(\d+)', url_or_id)
    if m:
        return m.group(1)

    # https://civitai.com/models/12345?modelVersionId=67890
    m = re.search(r'modelVersionId=(\d+)', url_or_id)
    if m:
        return m.group(1)

    # https://civitai.com/models/12345/name — need API call to get latest version
    m = re.search(r'civitai\.com/models/(\d+)', url_or_id)
    if m:
        import urllib.request
        import json as _json
        try:
            model_id = m.group(1)
            req = urllib.request.Request(f"{CIVITAI_API_BASE}/models/{model_id}")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
            versions = data.get("modelVersions", [])
            if versions:
                return str(versions[0]["id"])
        except Exception:
            pass

    return None


def _auto_populate_metadata(filename: str, lora_path: Path):
    """Read safetensors header and populate metadata."""
    try:
        from helpers import fast_safe_metadata
        meta = fast_safe_metadata(str(lora_path))
        update_lora_info(filename, {
            "rank": int(meta.get("rank", 0)),
            "alpha": int(meta.get("lora_alpha", 0)),
        })
    except Exception:
        pass
