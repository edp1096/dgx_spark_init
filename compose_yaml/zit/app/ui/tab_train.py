"""Train LoRA tab for ZIT UI."""

import logging
from pathlib import Path

import gradio as gr

from helpers import (
    DATASETS_BASE,
    scan_datasets, dataset_choices, create_dataset, upload_to_dataset,
    select_dataset_image, save_caption, delete_single_image,
    batch_prepend, batch_append, batch_delete_captions, auto_caption,
    dataset_contents,
)
from zit_config import MODEL_DIR

logger = logging.getLogger("zit-ui")

# ---------------------------------------------------------------------------
# Module-level mutable state (shared with restore function in app.py)
# ---------------------------------------------------------------------------
_active_trainer = {"mgr": None}
_train_ui_params = {"p": None}
_train_state = {
    "yield_active": False,
    "status": "Ready",
    "log": "",
    "progress_md": "Ready",
    "log_lines": [],
    "step": 0,
    "total": 0,
    "loss": 0.0,
    "elapsed": 0.0,
    "eta": 0.0,
    "msg": "",
}


def get_restore_train_params():
    """For app.load() — returns tuple of params or gr.update() skip."""
    skip = tuple([gr.update()] * 11)
    p = _train_ui_params.get("p")
    if not p or not _active_trainer.get("mgr"):
        return skip
    return (
        p["dataset"], p["name"],
        p["steps"], p["rank"], p["lr"], p["lora_alpha"],
        p["resolution"],
        p["batch"], p["grad_accum"], p["save_every"], p["targets"],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _fmt_time(secs):
    m, s = int(secs) // 60, int(secs) % 60
    return f"{m}:{s:02d}"


def _drain_train_queue():
    """Drain progress messages and update _train_state."""
    tmgr = _active_trainer.get("mgr")
    if tmgr is None:
        return
    ts = _train_state
    for msg in tmgr.poll_progress():
        if msg.get("type") == "status":
            ts["msg"] = msg["message"]
        elif msg.get("type") == "progress":
            ts["step"] = msg["step"]
            ts["total"] = msg["total"]
            ts["loss"] = msg["loss"]
            ts["elapsed"] = msg["elapsed"]
            ts["eta"] = msg["eta"]
            if ts["step"] % 50 == 0 or ts["step"] == 1:
                ts["log_lines"].append(
                    f"Step {ts['step']}/{ts['total']}  loss={ts['loss']:.4f}")

    # Build display strings
    if ts["step"] > 0 and ts["total"] > 0:
        pct = ts["step"] / ts["total"] * 100
        bar_len = 20
        filled = int(bar_len * ts["step"] / ts["total"])
        bar = "\u2593" * filled + "\u2591" * (bar_len - filled)
        ts["progress_md"] = (
            f"### Step {ts['step']} / {ts['total']} ({pct:.1f}%)\n"
            f"`{bar}`\n\n"
            f"**Loss:** {ts['loss']:.4f} | "
            f"**Elapsed:** {_fmt_time(ts['elapsed'])} | "
            f"**ETA:** {_fmt_time(ts['eta'])}"
        )
        ts["status"] = f"Training... step {ts['step']}/{ts['total']}"
    elif ts["msg"]:
        ts["progress_md"] = f"**{ts['msg']}**"
        ts["status"] = ts["msg"]
    ts["log"] = "\n".join(ts["log_lines"][-30:])

    # Check completion (process died)
    if not tmgr.is_alive():
        result = tmgr.get_result()
        if result is None:
            ts["status"] = "Training stopped"
            ts["progress_md"] = "**Stopped**"
        elif result["status"] == "done":
            output_path = result["path"]
            ts["status"] = f"Training complete! Saved: {Path(output_path).name}"
            ts["progress_md"] = (
                f"### Done\n**{Path(output_path).name}** | "
                f"{ts['step']} steps | {_fmt_time(ts['elapsed'])}"
            )
        else:
            tb = result.get("traceback", str(result.get("error", "Unknown")))
            ts["status"] = f"Error: {result.get('error', 'Unknown')}"
            ts["log"] = tb
            ts["progress_md"] = "**Failed**"
        _active_trainer["mgr"] = None
        _train_ui_params["p"] = None


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------
def build_train_tab(tab_ref):
    """Build Train tab UI, wire events, return components for app.load()."""

    def _get_train_status():
        if _active_trainer.get("mgr") and not _train_state["yield_active"]:
            _drain_train_queue()
        return _train_state["status"]

    def _get_train_log():
        return _train_state["log"]

    def _get_train_progress():
        if _active_trainer.get("mgr") and not _train_state["yield_active"]:
            _drain_train_queue()
        return _train_state["progress_md"]

    gr.Markdown("### LoRA Training")
    with gr.Row():
        with gr.Column(scale=1):
            # --- Dataset selector ---
            _ds_choices = dataset_choices()
            _ds_initial = _ds_choices[0] if _ds_choices else None
            _ds_gallery_init, _ds_summary_init = dataset_contents(_ds_initial) if _ds_initial else ([], "No dataset selected")
            with gr.Group():
                tr_dataset = gr.Dropdown(
                    choices=_ds_choices,
                    value=_ds_initial,
                    label="Dataset",
                    info="Select a dataset or create a new one below",
                    allow_custom_value=False,
                )
                tr_ds_summary = gr.Textbox(label="Dataset", interactive=False, lines=1, show_label=False, value=_ds_summary_init)
                with gr.Accordion("Data Images", open=True, elem_id="data-images-section"):
                    tr_ds_gallery_height = gr.State(200)
                    with gr.Row():
                        tr_ds_expand = gr.Button("Expand", size="sm", variant="secondary")
                    tr_ds_gallery = gr.Gallery(
                        label="Dataset Images (click to edit caption)", columns=4, height=200,
                        object_fit="cover", preview=False,
                        elem_id="dataset-gallery",
                        value=_ds_gallery_init, buttons=["download", "fullscreen"],
                    )

                    def _toggle_ds_gallery_height(current_h):
                        if current_h <= 200:
                            return 600, gr.Gallery(height=600), gr.Button(value="Collapse")
                        else:
                            return 200, gr.Gallery(height=200), gr.Button(value="Expand")
                    tr_ds_expand.click(
                        fn=_toggle_ds_gallery_height,
                        inputs=[tr_ds_gallery_height],
                        outputs=[tr_ds_gallery_height, tr_ds_gallery, tr_ds_expand],
                    )
                with gr.Accordion("Caption Editor", open=True):
                    tr_selected_image = gr.Textbox(visible=False)
                    tr_caption_edit = gr.Textbox(
                        label="Caption", lines=3,
                        placeholder="Click an image above to edit its caption",
                        interactive=True,
                    )
                    with gr.Row():
                        tr_save_caption = gr.Button("Save Caption", variant="primary", size="sm")
                        tr_delete_image = gr.Button("Delete Image", variant="stop", size="sm")
                    tr_caption_status = gr.Textbox(label="", interactive=False, lines=1, show_label=False)
                with gr.Accordion("Caption Tools", open=False):
                    tr_batch_text = gr.Textbox(
                        label="Keywords to add",
                        placeholder="e.g. a photo of sks person,",
                        lines=1,
                    )
                    with gr.Row():
                        tr_prepend_btn = gr.Button("Prepend to All", size="sm")
                        tr_append_btn = gr.Button("Append to All", size="sm")
                    gr.Markdown("---")
                    tr_trigger_word = gr.Textbox(
                        label="Trigger Word",
                        placeholder="e.g. lya",
                        info="Auto-prepended to every generated caption",
                        lines=1,
                    )
                    with gr.Row():
                        tr_autocap_btn = gr.Button("Auto-Caption (AI)", variant="primary", size="sm")
                        tr_autocap_overwrite = gr.Checkbox(label="Overwrite existing", value=False)
                    tr_autocap_status = gr.Textbox(label="Progress", interactive=False, lines=3)
                    gr.Markdown("---")
                    tr_delete_captions_btn = gr.Button("Delete All Captions", variant="stop", size="sm")
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
                tr_lora_alpha = gr.Number(value=16, label="LoRA Alpha", precision=0,
                                          minimum=1, maximum=128,
                                          info="PEFT scaling = alpha/rank (default=rank)")
            with gr.Row():
                tr_resolution = gr.Dropdown(
                    [256, 384, 512, 768, 1024], value=512, label="Resolution",
                )
            tr_batch = gr.Number(value=1, label="Batch Size", precision=0, minimum=1, maximum=32)
            tr_grad_accum = gr.Number(value=1, label="Gradient Accumulation", precision=0, minimum=1, maximum=8)
            tr_save_every = gr.Number(value=500, label="Save Checkpoint Every N Steps", precision=0)
            tr_targets = gr.Textbox(
                label="Target Modules",
                value="to_q, to_k, to_v, to_out.0, w1, w2, w3, adaLN_modulation.0",
                info="Comma-separated Linear layer names to train",
            )
            with gr.Accordion("Advanced", open=False):
                tr_use_deturbo = gr.Checkbox(label="Use De-Turbo model", value=True,
                                             info="Use de-distilled model for training")
                tr_caption_dropout = gr.Slider(0.0, 0.5, value=0.1, step=0.05,
                                               label="Caption Dropout")
                tr_noise_offset = gr.Slider(0.0, 0.1, value=0.0, step=0.01,
                                            label="Noise Offset")
                tr_diff_guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.5,
                                             label="Differential Guidance",
                                             info="0=off, 3.0=ostris default")
                tr_module_dropout = gr.Slider(0.0, 0.5, value=0.0, step=0.05,
                                              label="Module Dropout")
                tr_rank_dropout = gr.Slider(0.0, 0.5, value=0.0, step=0.05,
                                            label="Rank Dropout")
                tr_timestep_sampling = gr.Dropdown(
                    ["sigmoid", "uniform"], value="sigmoid",
                    label="Timestep Sampling",
                    info="sigmoid=중간 타임스텝 집중 (권장)",
                )
                tr_prefix_filter = gr.Dropdown(
                    ["layers.", ""], value="layers.",
                    label="Prefix Filter",
                    info="layers.=메인 블록만 (권장), 빈값=전체",
                )
            with gr.Row():
                tr_start = gr.Button("Start Training", variant="primary")
                tr_stop = gr.Button("Stop", variant="stop")
        with gr.Column(scale=1):
            tr_status = gr.Textbox(label="Status", interactive=False, lines=3,
                                   value=lambda: _get_train_status(), every=2)
            tr_log = gr.Textbox(label="Training Log", interactive=False, lines=15,
                                value=lambda: _get_train_log(), every=2)
            tr_progress = gr.Markdown(value=lambda: _get_train_progress(), every=1)

    # --- Tab select: auto-load first dataset ---
    def _on_train_tab():
        datasets = scan_datasets()
        if not datasets:
            return gr.update(choices=[], value=None), [], "No dataset selected"
        first = datasets[0]
        gallery, summary = dataset_contents(first)
        return gr.update(choices=datasets, value=first), gallery, summary

    tab_ref.select(
        fn=_on_train_tab,
        outputs=[tr_dataset, tr_ds_gallery, tr_ds_summary],
    )

    # --- Dataset management events ---
    tr_dataset.change(
        fn=dataset_contents,
        inputs=[tr_dataset],
        outputs=[tr_ds_gallery, tr_ds_summary],
    )
    tr_create_btn.click(
        fn=create_dataset,
        inputs=[tr_new_name],
        outputs=[tr_dataset, tr_new_name, tr_ds_status],
    )
    tr_upload.upload(
        fn=upload_to_dataset,
        inputs=[tr_upload, tr_dataset],
        outputs=[tr_ds_status, tr_ds_gallery, tr_ds_summary, tr_upload],
    )
    tr_ds_gallery.select(
        fn=select_dataset_image,
        inputs=[tr_dataset],
        outputs=[tr_selected_image, tr_caption_edit],
    )
    tr_save_caption.click(
        fn=save_caption,
        inputs=[tr_selected_image, tr_caption_edit, tr_dataset],
        outputs=[tr_caption_status, tr_ds_gallery, tr_ds_summary],
    )
    tr_delete_image.click(
        fn=delete_single_image,
        inputs=[tr_selected_image, tr_dataset],
        outputs=[tr_caption_status, tr_ds_gallery, tr_ds_summary, tr_selected_image, tr_caption_edit],
    )
    tr_prepend_btn.click(
        fn=batch_prepend,
        inputs=[tr_batch_text, tr_dataset],
        outputs=[tr_caption_status, tr_ds_gallery, tr_ds_summary],
    )
    tr_append_btn.click(
        fn=batch_append,
        inputs=[tr_batch_text, tr_dataset],
        outputs=[tr_caption_status, tr_ds_gallery, tr_ds_summary],
    )
    tr_autocap_btn.click(
        fn=auto_caption,
        inputs=[tr_dataset, tr_autocap_overwrite, tr_trigger_word],
        outputs=[tr_autocap_status, tr_ds_gallery, tr_ds_summary],
    )
    tr_delete_captions_btn.click(
        fn=batch_delete_captions,
        inputs=[tr_dataset],
        outputs=[tr_caption_status, tr_ds_gallery, tr_ds_summary],
    )

    # --- Training ---
    def _start_training(dataset_name, name, steps, rank, lr, lora_alpha,
                        resolution,
                        batch, grad_accum, save_every, targets,
                        use_deturbo, caption_dropout, noise_offset,
                        diff_guidance, module_dropout, rank_dropout,
                        timestep_sampling, prefix_filter):
        _train_ui_params["p"] = {
            "dataset": dataset_name, "name": name,
            "steps": steps, "rank": rank, "lr": lr,
            "lora_alpha": lora_alpha, "resolution": resolution,
            "batch": batch, "grad_accum": grad_accum,
            "save_every": save_every, "targets": targets,
        }
        ts = _train_state
        try:
            if not dataset_name:
                yield "Error: Select a dataset", "", "Error"
                return
            dataset = str(DATASETS_BASE / dataset_name)
            if not Path(dataset).is_dir():
                yield "Error: Dataset folder not found", "", "Error"
                return
            if not name:
                yield "Error: LoRA name required", "", "Error"
                return

            # Kill inference worker to free GPU
            from generators import get_worker_mgr
            wmgr = get_worker_mgr()
            if wmgr.is_alive():
                wmgr.kill()

            from trainer import TrainProcessManager
            tmgr = TrainProcessManager()
            _active_trainer["mgr"] = tmgr

            # Reset state
            ts["yield_active"] = True
            ts["step"] = 0
            ts["total"] = int(steps)
            ts["loss"] = 0.0
            ts["elapsed"] = 0.0
            ts["eta"] = 0.0
            ts["msg"] = "Preparing..."
            ts["log_lines"] = []
            ts["status"] = "Preparing..."
            ts["log"] = ""
            ts["progress_md"] = "**Preparing...**"

            target_list = [t.strip() for t in targets.split(",") if t.strip()]

            tmgr.start(
                model_dir=str(MODEL_DIR),
                dataset_dir=dataset,
                output_name=name,
                steps=int(steps),
                lr=float(lr),
                rank=int(rank),
                lora_alpha=int(lora_alpha),
                batch_size=int(batch),
                resolution=int(resolution),
                gradient_accumulation=int(grad_accum),
                save_every=int(save_every),
                target_modules=target_list,
                use_deturbo=bool(use_deturbo),
                caption_dropout=float(caption_dropout),
                noise_offset=float(noise_offset),
                differential_guidance=float(diff_guidance),
                module_dropout=float(module_dropout),
                rank_dropout=float(rank_dropout),
                timestep_sampling=str(timestep_sampling),
                prefix_filter=str(prefix_filter) if prefix_filter else None,
            )

            yield ts["status"], ts["log"], ts["progress_md"]

            import time as _time

            while tmgr.is_alive():
                _time.sleep(1.0)
                _drain_train_queue()
                yield ts["status"], ts["log"], ts["progress_md"]

            # Process finished — drain remaining messages
            _drain_train_queue()
            yield ts["status"], ts["log"], ts["progress_md"]

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            ts["status"] = f"Error: {e}"
            ts["log"] = tb
            ts["progress_md"] = "**Failed**"
            yield ts["status"], ts["log"], ts["progress_md"]
        finally:
            ts["yield_active"] = False
            # Do NOT kill training process here — it should continue
            # after browser refresh. Cleanup only if process already dead.
            mgr = _active_trainer.get("mgr")
            if mgr and not mgr.is_alive():
                _active_trainer["mgr"] = None

    def _stop_training():
        tmgr = _active_trainer.get("mgr")
        if tmgr and tmgr.is_alive():
            msg = tmgr.kill()
            _active_trainer["mgr"] = None
            _train_state["status"] = "Training stopped"
            _train_state["progress_md"] = "**Stopped**"
            return msg
        return "No training in progress"

    tr_start.click(
        fn=_start_training,
        inputs=[tr_dataset, tr_name, tr_steps, tr_rank, tr_lr, tr_lora_alpha,
                tr_resolution,
                tr_batch, tr_grad_accum, tr_save_every, tr_targets,
                tr_use_deturbo, tr_caption_dropout, tr_noise_offset,
                tr_diff_guidance, tr_module_dropout, tr_rank_dropout,
                tr_timestep_sampling, tr_prefix_filter],
        outputs=[tr_status, tr_log, tr_progress],
        concurrency_limit=1,
    )
    tr_stop.click(
        fn=_stop_training,
        inputs=[],
        outputs=[tr_status],
    )

    return {
        "dataset": tr_dataset, "name": tr_name,
        "steps": tr_steps, "rank": tr_rank, "lr": tr_lr, "lora_alpha": tr_lora_alpha,
        "resolution": tr_resolution,
        "batch": tr_batch, "grad_accum": tr_grad_accum,
        "save_every": tr_save_every, "targets": tr_targets,
    }
