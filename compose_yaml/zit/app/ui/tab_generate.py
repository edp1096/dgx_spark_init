"""Generate tab for ZIT UI."""

import gradio as gr

from generators import (
    generate_zit_t2i,
    generate_controlnet,
    get_gen_info_for_tab,
    get_loading_status,
    match_image_resolution,
    preview_preprocessor,
    save_gen_ui_params,
)
from helpers import (
    lora_choices, get_trigger_words, get_recommend_scale,
    do_kill, do_translate, translate_use,
    list_presets, load_preset_params, delete_preset,
    save_as_preset, export_preset, import_preset,
)
from translator import LANG_CHOICES, DEFAULT_LANG
from zit_config import (
    RESOLUTION_CHOICES, SAMPLE_PROMPTS, CONTROL_MODES,
    DEFAULT_STEPS, DEFAULT_TIME_SHIFT, DEFAULT_GUIDANCE,
    DEFAULT_CFG_TRUNCATION, DEFAULT_MAX_SEQ_LENGTH,
    MAX_LORA_STACK,
)

ATTENTION_BACKENDS = ["native", "flash", "flash_varlen", "_native_flash", "_native_math"]


def build_generate_tab():
    """Build Generate tab UI, wire events, return components for app.load()."""
    gr.Markdown("### Generate")
    with gr.Row():
        with gr.Column(scale=1):
            g_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your image...")
            with gr.Row():
                for i, sp in enumerate(SAMPLE_PROMPTS[:3]):
                    gr.Button(f"Sample {i+1}", size="sm", min_width=60).click(
                        fn=lambda s=sp: s, outputs=[g_prompt])
            with gr.Accordion("Translate", open=False):
                g_translate_target = gr.Radio(
                    ["Prompt", "Negative"], value="Prompt", label="Source", type="value",
                )
                with gr.Row():
                    g_translate_lang = gr.Dropdown(
                        choices=LANG_CHOICES, value=DEFAULT_LANG,
                        label="Target", scale=2, min_width=160,
                    )
                    g_translate_btn = gr.Button("Translate", size="sm", variant="secondary", scale=1)
                    g_translate_use = gr.Button("Use", size="sm", variant="secondary", scale=1)
                g_translate_result = gr.Textbox(label="Translation", lines=3, interactive=False)
            g_neg = gr.Textbox(label="Negative Prompt", lines=2)
            g_resolution = gr.Dropdown(
                RESOLUTION_CHOICES, value="512x768",
                label="Resolution (WxH)", allow_custom_value=True,
            )
            with gr.Row():
                g_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                g_num = gr.Number(value=1, label="Num Images", precision=0, minimum=1, maximum=32)
            g_steps = gr.Slider(1, 100, value=DEFAULT_STEPS, step=1, label="Steps")
            g_time_shift = gr.Slider(1.0, 12.0, value=DEFAULT_TIME_SHIFT, step=0.5, label="Time Shift")
            g_cfg = gr.Slider(0.0, 10.0, value=DEFAULT_GUIDANCE, step=0.5, label="Guidance Scale")
            g_cfg_norm = gr.Checkbox(label="CFG Normalization", value=False)
            g_cfg_trunc = gr.Slider(0.0, 1.0, value=DEFAULT_CFG_TRUNCATION, step=0.05, label="CFG Truncation")
            g_max_seq = gr.Slider(64, 1024, value=DEFAULT_MAX_SEQ_LENGTH, step=64, label="Max Sequence Length")
            g_use_fp8 = gr.Checkbox(label="FP8 Precision", value=True,
                info="FP8: fast+low VRAM / OFF: BF16 original quality (reload required)")
            g_attn = gr.Dropdown(
                ATTENTION_BACKENDS, value="native",
                label="Attention Backend",
                info="native=SDPA(auto FA2), flash=FA2, _native_flash=force SDPA flash",
            )
            with gr.Accordion("LoRA", open=False):
                g_lora_enable = gr.Checkbox(label="Enable LoRA", value=False)
                g_lora_count = gr.State(1)
                g_lora_rows = []
                g_lora_dropdowns = []
                g_lora_scales = []
                g_lora_triggers = []
                g_lora_remove_btns = []
                for i in range(MAX_LORA_STACK):
                    with gr.Row(visible=(i == 0)) as row:
                        dd = gr.Dropdown(lora_choices(), value="None", label=f"LoRA {i+1}",
                                         allow_custom_value=False, scale=3)
                        sl = gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="Scale", scale=1, min_width=80)
                        rm = gr.Button("✕", size="sm", variant="stop", scale=0, min_width=30,
                                       visible=(i > 0))
                    tw = gr.Textbox(label="", interactive=False, lines=1, max_lines=1,
                                    show_label=False, visible=(i == 0),
                                    placeholder="Trigger words")
                    g_lora_rows.append(row)
                    g_lora_dropdowns.append(dd)
                    g_lora_scales.append(sl)
                    g_lora_triggers.append(tw)
                    g_lora_remove_btns.append(rm)
                    def _on_lora_select_gen(name):
                        return get_trigger_words(name), get_recommend_scale(name)
                    dd.change(fn=_on_lora_select_gen, inputs=[dd], outputs=[tw, sl])
                with gr.Row():
                    g_lora_add = gr.Button("+ Add LoRA", size="sm", variant="secondary")
                    g_lora_refresh = gr.Button("Refresh", size="sm", variant="secondary")

                def _add_lora_slot(count):
                    count = min(count + 1, MAX_LORA_STACK)
                    row_vis = [gr.Row(visible=(i < count)) for i in range(MAX_LORA_STACK)]
                    tw_vis = [gr.Textbox(visible=(i < count)) for i in range(MAX_LORA_STACK)]
                    return [count] + row_vis + tw_vis

                g_lora_add.click(
                    fn=_add_lora_slot, inputs=[g_lora_count],
                    outputs=[g_lora_count] + g_lora_rows + g_lora_triggers,
                )

                def _remove_lora_slot(count, idx):
                    count = max(count - 1, 1)
                    row_vis = [gr.Row(visible=(i < count)) for i in range(MAX_LORA_STACK)]
                    tw_vis = [gr.Textbox(visible=(i < count)) for i in range(MAX_LORA_STACK)]
                    dd_updates = [gr.update()] * MAX_LORA_STACK
                    dd_updates[idx] = gr.Dropdown(value="None")
                    return [count] + row_vis + tw_vis + dd_updates

                for idx, rm_btn in enumerate(g_lora_remove_btns):
                    rm_btn.click(
                        fn=lambda cnt, i=idx: _remove_lora_slot(cnt, i),
                        inputs=[g_lora_count],
                        outputs=[g_lora_count] + g_lora_rows + g_lora_triggers + g_lora_dropdowns,
                    )

                def _refresh_all_loras():
                    choices = lora_choices()
                    return [gr.Dropdown(choices=choices)] * MAX_LORA_STACK

                g_lora_refresh.click(fn=_refresh_all_loras, outputs=g_lora_dropdowns)
            g_cn_enable = gr.Checkbox(
                label="Enable ControlNet", value=False,
                info="ON: load ControlNet adapter (pose/depth control) / OFF: pure T2I (better face quality)",
            )
            g_cn_scale = gr.Slider(
                0.0, 1.0, value=0.65, step=0.05, label="Control Scale",
                visible=False,
            )
            g_generate = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            # --- ControlNet controls (visible when enabled) ---
            g_cn_panel = gr.Group(visible=False)
            with g_cn_panel:
                g_cn_mode = gr.Radio(
                    CONTROL_MODES, value="canny", label="Control Mode",
                )
                g_cn_image = gr.Image(label="Control Image", type="numpy", buttons=["download", "fullscreen"])
                with gr.Row():
                    g_cn_preview_btn = gr.Button(
                        "Preview Preprocessor", variant="secondary", size="sm",
                    )
                    g_cn_match_res = gr.Button(
                        "Match Image Size", size="sm", variant="secondary",
                    )
                g_cn_preview = gr.Image(
                    label="Control Preview", interactive=False,
                    buttons=["download", "fullscreen"],
                )

            # --- Presets ---
            with gr.Accordion("Presets", open=False, elem_id="presets-section"):
                preset_height = gr.State(200)
                preset_sel_idx = gr.State(-1)
                with gr.Row(elem_id="presets-toggle-row"):
                    g_preset_expand = gr.Button("Expand", size="sm", variant="secondary")
                    g_save_preset = gr.Button("Save as Preset", size="sm", variant="primary")
                preset_gallery = gr.Gallery(
                    label="Click to load preset",
                    value=list_presets,
                    columns=3, height=200, object_fit="contain",
                    preview=False, elem_id="presets-gallery",
                )
                g_save_status = gr.Textbox(label="", interactive=False, lines=1, show_label=False)
                g_delete_preset = gr.Button("Delete Selected Preset", size="sm", variant="stop")
                gr.Markdown("### JSON")
                with gr.Row():
                    g_preset_export = gr.Button("Export JSON", size="sm", variant="secondary")
                    g_preset_import = gr.UploadButton("Import JSON", size="sm", variant="secondary", file_types=[".json"])
                g_preset_download = gr.File(visible=False)

                def _toggle_preset_height(current_h):
                    if current_h <= 200:
                        return 600, gr.Gallery(height=600), gr.Button(value="Collapse")
                    else:
                        return 200, gr.Gallery(height=200), gr.Button(value="Expand")
                g_preset_expand.click(
                    fn=_toggle_preset_height,
                    inputs=[preset_height],
                    outputs=[preset_height, preset_gallery, g_preset_expand],
                )

            # --- Gallery & status ---
            g_gallery = gr.Gallery(label="Generated Images", columns=2, height=500, object_fit="contain", elem_id="gen-gallery", preview=True, selected_index=0, buttons=["download", "fullscreen"])
            g_info = gr.Textbox(label="Info", interactive=False,
                                value=lambda: get_gen_info_for_tab("generate"), every=2)
            with gr.Row():
                g_kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm")
            g_kill_msg = gr.Textbox(label="", interactive=False, visible=False)
            gr.Markdown(value=get_loading_status, every=1)
            g_kill_btn.click(fn=do_kill, outputs=[g_kill_msg])
            g_gen_paths = gr.State([])

    # --- ControlNet toggle: show/hide panel + scale ---
    def _toggle_cn(enabled):
        return gr.update(visible=enabled), gr.update(visible=enabled)

    g_cn_enable.change(
        fn=_toggle_cn,
        inputs=[g_cn_enable],
        outputs=[g_cn_panel, g_cn_scale],
    )
    g_cn_preview_btn.click(
        fn=lambda img, mode: preview_preprocessor(mode, img),
        inputs=[g_cn_image, g_cn_mode],
        outputs=[g_cn_preview],
        concurrency_limit=1,
    )
    g_cn_match_res.click(
        fn=match_image_resolution,
        inputs=[g_cn_image],
        outputs=[g_resolution],
    )

    # --- Generate dispatch ---
    def _build_lora_stack(enable, count, *dd_and_scales):
        """Collect active LoRA entries into a stack list."""
        if not enable:
            return []
        stack = []
        count = int(count)
        for i in range(min(count, MAX_LORA_STACK)):
            name = dd_and_scales[i]
            scale = dd_and_scales[MAX_LORA_STACK + i]
            if name and name != "None":
                stack.append({"name": name, "scale": float(scale)})
        return stack

    def _generate_dispatch(prompt, resolution, seed, num_images,
                           neg, steps, time_shift, cfg, cfg_norm, cfg_trunc,
                           max_seq, use_fp8, attn_backend,
                           lora_enable, lora_count,
                           *lora_and_cn_args,
                           progress=gr.Progress(track_tqdm=True)):
        # Unpack: MAX_LORA_STACK dropdowns + MAX_LORA_STACK scales + cn args
        lora_dds = list(lora_and_cn_args[:MAX_LORA_STACK])
        lora_sls = list(lora_and_cn_args[MAX_LORA_STACK:MAX_LORA_STACK*2])
        cn_enable, cn_mode, cn_image, cn_scale = lora_and_cn_args[MAX_LORA_STACK*2:]

        lora_stack = _build_lora_stack(lora_enable, lora_count, *lora_dds, *lora_sls)

        save_gen_ui_params({
            "tab": "generate",
            "prompt": prompt, "neg": neg, "resolution": resolution,
            "seed": seed, "num_images": num_images,
            "steps": steps, "time_shift": time_shift,
            "cfg": cfg, "cfg_norm": cfg_norm, "cfg_trunc": cfg_trunc,
            "max_seq": max_seq, "use_fp8": use_fp8, "attn": attn_backend,
            "lora_enable": lora_enable, "lora_stack": lora_stack,
            "cn_enable": cn_enable, "cn_mode": cn_mode, "cn_scale": cn_scale,
        })

        if cn_enable:
            if cn_image is None:
                raise gr.Error("ControlNet is enabled but no control image uploaded.")
            preprocessed = preview_preprocessor(cn_mode, cn_image)
            paths, info = generate_controlnet(
                prompt, cn_mode, preprocessed, resolution, seed,
                negative_prompt=neg, num_steps=steps,
                guidance_scale=cfg, cfg_normalization=cfg_norm,
                cfg_truncation=cfg_trunc, control_scale=cn_scale,
                max_sequence_length=max_seq, time_shift=time_shift,
                num_images=num_images, attention_backend=attn_backend,
                lora_stack=lora_stack,
                use_fp8=use_fp8,
                progress=progress,
            )
        else:
            paths, info = generate_zit_t2i(
                prompt, resolution, seed, num_images,
                negative_prompt=neg, num_steps=steps,
                time_shift=time_shift,
                guidance_scale=cfg,
                cfg_normalization=cfg_norm, cfg_truncation=cfg_trunc,
                max_sequence_length=max_seq,
                attention_backend=attn_backend,
                lora_stack=lora_stack,
                use_fp8=use_fp8,
                progress=progress,
            )
        return gr.Gallery(value=paths, selected_index=0), info, paths

    g_generate.click(
        fn=_generate_dispatch,
        inputs=[g_prompt, g_resolution, g_seed, g_num,
                g_neg, g_steps, g_time_shift, g_cfg, g_cfg_norm, g_cfg_trunc,
                g_max_seq, g_use_fp8, g_attn,
                g_lora_enable, g_lora_count,
                *g_lora_dropdowns, *g_lora_scales,
                g_cn_enable, g_cn_mode, g_cn_image, g_cn_scale],
        outputs=[g_gallery, g_info, g_gen_paths],
        concurrency_limit=1,
    )

    # --- Presets ---
    preset_gallery.select(
        fn=load_preset_params,
        outputs=[g_prompt, g_neg, g_resolution, g_seed,
                 g_steps, g_time_shift, g_cfg, g_cfg_norm, g_cfg_trunc, g_max_seq,
                 preset_sel_idx],
    )
    g_save_preset.click(
        fn=save_as_preset,
        inputs=[g_gen_paths, g_prompt, g_neg, g_resolution, g_seed,
                g_steps, g_time_shift, g_cfg, g_cfg_norm, g_cfg_trunc, g_max_seq],
        outputs=[preset_gallery, g_save_status],
    )
    g_delete_preset.click(
        fn=delete_preset,
        inputs=[preset_sel_idx],
        outputs=[preset_gallery, g_save_status],
    )
    g_preset_export.click(
        fn=export_preset,
        inputs=[g_prompt, g_neg, g_resolution, g_seed,
                g_steps, g_time_shift, g_cfg, g_cfg_norm, g_cfg_trunc, g_max_seq],
        outputs=[g_preset_download],
    )
    g_preset_import.upload(
        fn=import_preset,
        inputs=[g_preset_import],
        outputs=[g_prompt, g_neg, g_resolution, g_seed,
                 g_steps, g_time_shift, g_cfg, g_cfg_norm, g_cfg_trunc, g_max_seq],
    )

    # --- Translate ---
    def _g_translate(prompt, neg, target_sel, lang):
        src = prompt if target_sel == "Prompt" else neg
        return do_translate(src, lang)

    g_translate_btn.click(
        fn=_g_translate,
        inputs=[g_prompt, g_neg, g_translate_target, g_translate_lang],
        outputs=[g_translate_result],
    )
    g_translate_use.click(
        fn=translate_use,
        inputs=[g_translate_result, g_prompt, g_neg, g_translate_target],
        outputs=[g_prompt, g_neg],
    )

    return {
        "prompt": g_prompt, "neg": g_neg, "resolution": g_resolution,
        "seed": g_seed, "num_images": g_num,
        "steps": g_steps, "time_shift": g_time_shift,
        "cfg": g_cfg, "cfg_norm": g_cfg_norm, "cfg_trunc": g_cfg_trunc,
        "max_seq": g_max_seq, "use_fp8": g_use_fp8, "attn": g_attn,
        "lora_enable": g_lora_enable,
        "lora_dropdowns": g_lora_dropdowns, "lora_scales": g_lora_scales,
        "cn_enable": g_cn_enable, "cn_mode": g_cn_mode,
        "cn_image": g_cn_image, "cn_scale": g_cn_scale,
        "gallery": g_gallery,
    }
