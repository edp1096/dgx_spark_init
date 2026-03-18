"""Inpaint / Outpaint tab for ZIT UI."""

import gradio as gr

from generators import (
    generate_inpaint,
    generate_outpaint,
    get_gen_info_for_tab,
    get_loading_status,
    save_gen_ui_params,
)
from helpers import lora_choices, do_kill, do_translate, translate_use
from translator import LANG_CHOICES, DEFAULT_LANG
from zit_config import (
    RESOLUTION_CHOICES,
    DEFAULT_INPAINT_STEPS,
    DEFAULT_TIME_SHIFT,
    DEFAULT_INPAINT_GUIDANCE,
    DEFAULT_INPAINT_CFG_TRUNCATION,
    DEFAULT_INPAINT_CONTROL_SCALE,
    DEFAULT_MAX_SEQ_LENGTH,
)


def build_inpaint_tab():
    """Build Inpaint/Outpaint tab UI, wire events, return components for app.load()."""
    gr.Markdown("### Inpaint / Outpaint")
    with gr.Row():
        with gr.Column(scale=1):
            ip_mode = gr.Radio(["Inpaint", "Outpaint"], value="Inpaint", label="Mode")

            # Inpaint controls
            ip_editor = gr.ImageEditor(
                label="Draw Mask (white = regenerate)",
                type="numpy",
                image_mode="RGB",
                brush=gr.Brush(colors=["#ffffff"], default_size=20),
                eraser=gr.Eraser(default_size=20),
            )

            # Outpaint controls
            ip_out_image = gr.Image(label="Image", type="numpy", visible=False)
            ip_direction = gr.CheckboxGroup(
                ["Left", "Right", "Up", "Down"],
                value=["Right"], label="Expand Direction", visible=False,
            )
            ip_expand = gr.Slider(64, 512, value=256, step=64, label="Expand Size (px)", visible=False)

            ip_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe what to fill...")
            with gr.Accordion("Translate", open=False):
                ip_translate_target = gr.Radio(
                    ["Prompt", "Negative"], value="Prompt", label="Source", type="value",
                )
                with gr.Row():
                    ip_translate_lang = gr.Dropdown(
                        choices=LANG_CHOICES, value=DEFAULT_LANG,
                        label="Target", scale=2, min_width=160,
                    )
                    ip_translate_btn = gr.Button("Translate", size="sm", variant="secondary", scale=1)
                    ip_translate_use = gr.Button("Use", size="sm", variant="secondary", scale=1)
                ip_translate_result = gr.Textbox(label="Translation", lines=3, interactive=False)
            ip_neg = gr.Textbox(label="Negative Prompt", lines=2)
            ip_resolution = gr.Dropdown(
                RESOLUTION_CHOICES, value="512x768",
                label="Resolution (WxH)", allow_custom_value=True,
            )
            ip_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
            ip_steps = gr.Slider(1, 100, value=DEFAULT_INPAINT_STEPS, step=1, label="Steps")
            ip_time_shift = gr.Slider(1.0, 12.0, value=DEFAULT_TIME_SHIFT, step=0.5, label="Time Shift")
            ip_control_scale = gr.Slider(0.0, 1.0, value=DEFAULT_INPAINT_CONTROL_SCALE, step=0.05, label="Control Scale")
            ip_guidance = gr.Slider(0.0, 10.0, value=DEFAULT_INPAINT_GUIDANCE, step=0.5, label="Guidance Scale")
            ip_cfg_trunc = gr.Slider(0.0, 1.0, value=DEFAULT_INPAINT_CFG_TRUNCATION, step=0.05, label="CFG Truncation")
            ip_max_seq = gr.Slider(64, 1024, value=DEFAULT_MAX_SEQ_LENGTH, step=64, label="Max Sequence Length")
            ip_use_controlnet = gr.Checkbox(label="Enable ControlNet", value=True)
            with gr.Accordion("LoRA", open=False):
                ip_lora_enable = gr.Checkbox(label="Enable LoRA", value=False)
                ip_lora = gr.Dropdown(
                    lora_choices(), value="None", label="LoRA",
                    allow_custom_value=False,
                )
                ip_lora_scale = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="LoRA Scale")
                ip_lora_refresh = gr.Button("Refresh", size="sm", variant="secondary")
                ip_lora_refresh.click(
                    fn=lambda: gr.Dropdown(choices=lora_choices(), value="None"),
                    outputs=[ip_lora],
                )
            ip_gen_inpaint = gr.Button("Generate", variant="primary", visible=True)
            ip_gen_outpaint = gr.Button("Generate", variant="primary", visible=False)

        with gr.Column(scale=1):
            ip_result = gr.Image(label="Result", type="filepath", buttons=["download", "fullscreen"])
            ip_info = gr.Textbox(label="Info", interactive=False,
                                 value=lambda: get_gen_info_for_tab("inpaint"), every=2)
            ip_kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm")
            ip_kill_msg = gr.Textbox(label="", interactive=False, visible=False)
            gr.Markdown(value=get_loading_status, every=1)
            ip_kill_btn.click(fn=do_kill, outputs=[ip_kill_msg])

    # --- Mode switch ---
    def _on_ip_mode(mode):
        is_inpaint = mode == "Inpaint"
        return [
            gr.ImageEditor(visible=is_inpaint),
            gr.Image(visible=not is_inpaint),
            gr.CheckboxGroup(visible=not is_inpaint),
            gr.Slider(visible=not is_inpaint),
            gr.Button(visible=is_inpaint),
            gr.Button(visible=not is_inpaint),
        ]

    ip_mode.change(
        fn=_on_ip_mode, inputs=[ip_mode],
        outputs=[ip_editor, ip_out_image, ip_direction, ip_expand,
                 ip_gen_inpaint, ip_gen_outpaint],
    )

    # --- Translate ---
    def _ip_translate(prompt, neg, target_sel, lang):
        src = prompt if target_sel == "Prompt" else neg
        return do_translate(src, lang)

    ip_translate_btn.click(
        fn=_ip_translate,
        inputs=[ip_prompt, ip_neg, ip_translate_target, ip_translate_lang],
        outputs=[ip_translate_result],
    )
    ip_translate_use.click(
        fn=translate_use,
        inputs=[ip_translate_result, ip_prompt, ip_neg, ip_translate_target],
        outputs=[ip_prompt, ip_neg],
    )

    # --- Generate inpaint ---
    def _do_inpaint(editor_val, prompt, neg, resolution, seed,
                    steps, time_shift, control_scale, guidance, cfg_trunc, max_seq,
                    use_controlnet, lora_enable, lora, lora_scale,
                    progress=gr.Progress(track_tqdm=True)):
        save_gen_ui_params({
            "tab": "inpaint",
            "prompt": prompt, "neg": neg, "resolution": resolution,
            "seed": seed,
            "steps": steps, "time_shift": time_shift,
            "control_scale": control_scale, "guidance": guidance,
            "cfg_trunc": cfg_trunc, "max_seq": max_seq,
            "use_controlnet": use_controlnet,
            "lora_enable": lora_enable, "lora": lora, "lora_scale": lora_scale,
        })
        effective_lora = lora if lora_enable and lora != "None" else None
        paths, info = generate_inpaint(
            prompt, editor_val, resolution, seed,
            negative_prompt=neg, num_steps=steps, guidance_scale=guidance,
            cfg_truncation=cfg_trunc, control_scale=control_scale,
            max_sequence_length=max_seq, time_shift=time_shift,
            lora_name=effective_lora, lora_scale=lora_scale,
            need_controlnet=use_controlnet,
            progress=progress,
        )
        return paths[0] if paths else None, info

    ip_gen_inpaint.click(
        fn=_do_inpaint,
        inputs=[ip_editor,
                ip_prompt, ip_neg, ip_resolution, ip_seed,
                ip_steps, ip_time_shift, ip_control_scale, ip_guidance, ip_cfg_trunc, ip_max_seq,
                ip_use_controlnet, ip_lora_enable, ip_lora, ip_lora_scale],
        outputs=[ip_result, ip_info],
        concurrency_limit=1,
    )

    # --- Generate outpaint ---
    def _do_outpaint(out_image, direction, expand_px,
                     prompt, neg, resolution, seed,
                     steps, time_shift, control_scale, guidance, cfg_trunc, max_seq,
                     use_controlnet, lora_enable, lora, lora_scale,
                     progress=gr.Progress(track_tqdm=True)):
        save_gen_ui_params({
            "tab": "inpaint",
            "prompt": prompt, "neg": neg, "resolution": resolution,
            "seed": seed,
            "steps": steps, "time_shift": time_shift,
            "control_scale": control_scale, "guidance": guidance,
            "cfg_trunc": cfg_trunc, "max_seq": max_seq,
            "use_controlnet": use_controlnet,
            "lora_enable": lora_enable, "lora": lora, "lora_scale": lora_scale,
        })
        effective_lora = lora if lora_enable and lora != "None" else None
        paths, info = generate_outpaint(
            prompt, out_image, direction, expand_px, resolution, seed,
            negative_prompt=neg, num_steps=steps, guidance_scale=guidance,
            cfg_truncation=cfg_trunc, control_scale=control_scale,
            max_sequence_length=max_seq, time_shift=time_shift,
            lora_name=effective_lora, lora_scale=lora_scale,
            need_controlnet=use_controlnet,
            progress=progress,
        )
        return paths[0] if paths else None, info

    ip_gen_outpaint.click(
        fn=_do_outpaint,
        inputs=[ip_out_image, ip_direction, ip_expand,
                ip_prompt, ip_neg, ip_resolution, ip_seed,
                ip_steps, ip_time_shift, ip_control_scale, ip_guidance, ip_cfg_trunc, ip_max_seq,
                ip_use_controlnet, ip_lora_enable, ip_lora, ip_lora_scale],
        outputs=[ip_result, ip_info],
        concurrency_limit=1,
    )

    return {
        "prompt": ip_prompt, "neg": ip_neg, "resolution": ip_resolution,
        "seed": ip_seed,
        "steps": ip_steps, "time_shift": ip_time_shift,
        "control_scale": ip_control_scale, "guidance": ip_guidance,
        "cfg_trunc": ip_cfg_trunc, "max_seq": ip_max_seq,
        "use_controlnet": ip_use_controlnet,
        "lora_enable": ip_lora_enable, "lora": ip_lora, "lora_scale": ip_lora_scale,
        "result": ip_result,
    }
