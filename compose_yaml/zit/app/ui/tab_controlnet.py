"""ControlNet tab for ZIT UI."""

import gradio as gr

from generators import (
    generate_controlnet,
    get_gen_info_for_tab,
    get_loading_status,
    match_image_resolution,
    preview_preprocessor,
    save_gen_ui_params,
)
from helpers import lora_choices, do_kill, do_translate, translate_use
from translator import LANG_CHOICES, DEFAULT_LANG
from zit_config import (
    RESOLUTION_CHOICES, CONTROL_MODES,
    DEFAULT_STEPS, DEFAULT_TIME_SHIFT, DEFAULT_GUIDANCE,
    DEFAULT_CFG_TRUNCATION, DEFAULT_MAX_SEQ_LENGTH,
)


def build_controlnet_tab():
    """Build ControlNet tab UI, wire events, return components for app.load()."""
    gr.Markdown("### ControlNet")
    with gr.Row():
        with gr.Column(scale=1):
            cn_mode = gr.Radio(
                CONTROL_MODES, value="canny", label="Control Mode",
            )
            cn_image = gr.Image(label="Input Image", type="numpy")
            cn_preview_btn = gr.Button("Preview Preprocessor", variant="secondary", size="sm")
            cn_preview = gr.Image(label="Control Preview", interactive=False, buttons=["download", "fullscreen"])
            cn_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe your image...")
            with gr.Accordion("Translate", open=False):
                cn_translate_target = gr.Radio(
                    ["Prompt", "Negative"], value="Prompt", label="Source", type="value",
                )
                with gr.Row():
                    cn_translate_lang = gr.Dropdown(
                        choices=LANG_CHOICES, value=DEFAULT_LANG,
                        label="Target", scale=2, min_width=160,
                    )
                    cn_translate_btn = gr.Button("Translate", size="sm", variant="secondary", scale=1)
                    cn_translate_use = gr.Button("Use", size="sm", variant="secondary", scale=1)
                cn_translate_result = gr.Textbox(label="Translation", lines=3, interactive=False)
            cn_neg = gr.Textbox(label="Negative Prompt", lines=2)
            cn_resolution = gr.Dropdown(
                RESOLUTION_CHOICES, value="512x768",
                label="Resolution (WxH)", allow_custom_value=True,
            )
            cn_match_res = gr.Button("Match Image Size", size="sm", variant="secondary")
            cn_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
            cn_steps = gr.Slider(1, 100, value=DEFAULT_STEPS, step=1, label="Steps")
            cn_time_shift = gr.Slider(1.0, 12.0, value=DEFAULT_TIME_SHIFT, step=0.5, label="Time Shift")
            cn_control_scale = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="Control Scale")
            cn_guidance = gr.Slider(0.0, 10.0, value=DEFAULT_GUIDANCE, step=0.5, label="Guidance Scale")
            cn_cfg_trunc = gr.Slider(0.0, 1.0, value=DEFAULT_CFG_TRUNCATION, step=0.05, label="CFG Truncation")
            cn_max_seq = gr.Slider(64, 1024, value=DEFAULT_MAX_SEQ_LENGTH, step=64, label="Max Sequence Length")
            cn_use_fp8 = gr.Checkbox(label="FP8 Precision", value=True,
                info="FP8: fast+low VRAM / OFF: BF16 original quality")
            with gr.Accordion("LoRA", open=False):
                cn_lora = gr.Dropdown(
                    lora_choices(), value="None", label="LoRA",
                    allow_custom_value=False,
                )
                cn_lora_scale = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="LoRA Scale")
                cn_lora_refresh = gr.Button("Refresh", size="sm", variant="secondary")
                cn_lora_refresh.click(
                    fn=lambda: gr.Dropdown(choices=lora_choices(), value="None"),
                    outputs=[cn_lora],
                )
            cn_generate = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            cn_gallery = gr.Gallery(label="Generated Images", columns=2, height=500, object_fit="contain", elem_id="cn-gallery", preview=True, selected_index=0)
            cn_info = gr.Textbox(label="Info", interactive=False,
                                 value=lambda: get_gen_info_for_tab("controlnet"), every=2)
            cn_kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm")
            cn_kill_msg = gr.Textbox(label="", interactive=False, visible=False)
            gr.Markdown(value=get_loading_status, every=1)
            cn_kill_btn.click(fn=do_kill, outputs=[cn_kill_msg])

    # --- Preview preprocessor ---
    cn_preview_btn.click(
        fn=lambda img, mode: preview_preprocessor(mode, img),
        inputs=[cn_image, cn_mode],
        outputs=[cn_preview],
        concurrency_limit=1,
    )

    # --- Match image size ---
    cn_match_res.click(
        fn=match_image_resolution,
        inputs=[cn_image],
        outputs=[cn_resolution],
    )

    # --- Generate with ControlNet ---
    def _cn_generate(mode, prompt, neg, image, resolution, seed,
                     steps, time_shift, control_scale, guidance, cfg_trunc, max_seq,
                     use_fp8, lora, lora_scale,
                     progress=gr.Progress(track_tqdm=True)):
        save_gen_ui_params({
            "tab": "controlnet",
            "prompt": prompt, "neg": neg, "resolution": resolution,
            "seed": seed, "mode": mode,
            "steps": steps, "time_shift": time_shift,
            "control_scale": control_scale, "guidance": guidance,
            "cfg_trunc": cfg_trunc, "max_seq": max_seq,
            "use_fp8": use_fp8, "lora": lora, "lora_scale": lora_scale,
        })
        preprocessed = preview_preprocessor(mode, image)
        effective_lora = lora if lora != "None" else None
        paths, info = generate_controlnet(
            prompt, mode, preprocessed, resolution, seed,
            negative_prompt=neg, num_steps=steps, guidance_scale=guidance,
            cfg_truncation=cfg_trunc, control_scale=control_scale,
            max_sequence_length=max_seq, time_shift=time_shift,
            lora_name=effective_lora,
            lora_scale=lora_scale,
            use_fp8=use_fp8,
            progress=progress,
        )
        return gr.Gallery(value=paths, selected_index=0), info

    cn_generate.click(
        fn=_cn_generate,
        inputs=[cn_mode, cn_prompt, cn_neg, cn_image, cn_resolution, cn_seed,
                cn_steps, cn_time_shift, cn_control_scale, cn_guidance, cn_cfg_trunc, cn_max_seq,
                cn_use_fp8, cn_lora, cn_lora_scale],
        outputs=[cn_gallery, cn_info],
        concurrency_limit=1,
    )

    # --- Translate ---
    def _cn_translate(prompt, neg, target_sel, lang):
        src = prompt if target_sel == "Prompt" else neg
        return do_translate(src, lang)

    cn_translate_btn.click(
        fn=_cn_translate,
        inputs=[cn_prompt, cn_neg, cn_translate_target, cn_translate_lang],
        outputs=[cn_translate_result],
    )
    cn_translate_use.click(
        fn=translate_use,
        inputs=[cn_translate_result, cn_prompt, cn_neg, cn_translate_target],
        outputs=[cn_prompt, cn_neg],
    )

    return {
        "prompt": cn_prompt, "neg": cn_neg, "resolution": cn_resolution,
        "seed": cn_seed, "mode": cn_mode,
        "steps": cn_steps, "time_shift": cn_time_shift,
        "control_scale": cn_control_scale, "guidance": cn_guidance,
        "cfg_trunc": cn_cfg_trunc, "max_seq": cn_max_seq, "use_fp8": cn_use_fp8,
        "lora": cn_lora, "lora_scale": cn_lora_scale,
        "gallery": cn_gallery,
    }
