"""Inpaint / Outpaint tab for ZIT UI."""

import gradio as gr

from generators import (
    generate_inpaint,
    generate_outpaint,
    get_gen_info_for_tab,
    get_loading_status,
    save_gen_ui_params,
)
from helpers import lora_choices, get_trigger_words, get_recommend_scale, do_kill, do_translate, translate_use
from translator import LANG_CHOICES, DEFAULT_LANG
from zit_config import (
    RESOLUTION_CHOICES,
    DEFAULT_INPAINT_STEPS,
    DEFAULT_TIME_SHIFT,
    DEFAULT_INPAINT_GUIDANCE,
    DEFAULT_INPAINT_CFG_TRUNCATION,
    DEFAULT_INPAINT_CONTROL_SCALE,
    DEFAULT_MAX_SEQ_LENGTH,
    MAX_LORA_STACK,
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
                ip_lora_count = gr.State(1)
                ip_lora_rows = []
                ip_lora_dropdowns = []
                ip_lora_scales = []
                ip_lora_triggers = []
                ip_lora_remove_btns = []
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
                    ip_lora_rows.append(row)
                    ip_lora_dropdowns.append(dd)
                    ip_lora_scales.append(sl)
                    ip_lora_triggers.append(tw)
                    ip_lora_remove_btns.append(rm)
                    def _on_lora_select_ip(name):
                        return get_trigger_words(name), get_recommend_scale(name)
                    dd.change(fn=_on_lora_select_ip, inputs=[dd], outputs=[tw, sl])
                with gr.Row():
                    ip_lora_add = gr.Button("+ Add LoRA", size="sm", variant="secondary")
                    ip_lora_refresh = gr.Button("Refresh", size="sm", variant="secondary")

                def _ip_add_lora(count):
                    count = min(count + 1, MAX_LORA_STACK)
                    row_vis = [gr.update(visible=(i < count)) for i in range(MAX_LORA_STACK)]
                    tw_vis = [gr.update(visible=(i < count)) for i in range(MAX_LORA_STACK)]
                    return [count] + row_vis + tw_vis

                ip_lora_add.click(
                    fn=_ip_add_lora, inputs=[ip_lora_count],
                    outputs=[ip_lora_count] + ip_lora_rows + ip_lora_triggers,
                )

                def _ip_remove_lora(count, idx, *all_dd_and_tw):
                    """Remove slot idx, shift subsequent slots up."""
                    dds = list(all_dd_and_tw[:MAX_LORA_STACK])
                    tws = list(all_dd_and_tw[MAX_LORA_STACK:])
                    for j in range(idx, count - 1):
                        if j + 1 < MAX_LORA_STACK:
                            dds[j] = dds[j + 1]
                            tws[j] = tws[j + 1]
                    last = count - 1
                    if last < MAX_LORA_STACK:
                        dds[last] = "None"
                        tws[last] = ""
                    count = max(count - 1, 1)
                    row_vis = [gr.update(visible=(i < count)) for i in range(MAX_LORA_STACK)]
                    tw_updates = [gr.update(visible=(i < count), value=tws[i]) for i in range(MAX_LORA_STACK)]
                    dd_updates = [gr.update(value=dds[i]) for i in range(MAX_LORA_STACK)]
                    return [count] + row_vis + tw_updates + dd_updates

                for idx, rm_btn in enumerate(ip_lora_remove_btns):
                    rm_btn.click(
                        fn=lambda cnt, *args, i=idx: _ip_remove_lora(cnt, i, *args),
                        inputs=[ip_lora_count] + ip_lora_dropdowns + ip_lora_triggers,
                        outputs=[ip_lora_count] + ip_lora_rows + ip_lora_triggers + ip_lora_dropdowns,
                    )

                def _ip_refresh_loras():
                    choices = lora_choices()
                    return [gr.Dropdown(choices=choices)] * MAX_LORA_STACK

                ip_lora_refresh.click(fn=_ip_refresh_loras, outputs=ip_lora_dropdowns)
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

    # --- Helper: build LoRA stack ---
    def _ip_build_lora_stack(enable, count, *dd_and_scales):
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

    # --- Generate inpaint ---
    def _do_inpaint(editor_val, prompt, neg, resolution, seed,
                    steps, time_shift, control_scale, guidance, cfg_trunc, max_seq,
                    use_controlnet, lora_enable, lora_count,
                    *lora_args,
                    progress=gr.Progress(track_tqdm=True)):
        lora_dds = list(lora_args[:MAX_LORA_STACK])
        lora_sls = list(lora_args[MAX_LORA_STACK:])
        lora_stack = _ip_build_lora_stack(lora_enable, lora_count, *lora_dds, *lora_sls)
        save_gen_ui_params({
            "tab": "inpaint",
            "prompt": prompt, "neg": neg, "resolution": resolution,
            "seed": seed,
            "steps": steps, "time_shift": time_shift,
            "control_scale": control_scale, "guidance": guidance,
            "cfg_trunc": cfg_trunc, "max_seq": max_seq,
            "use_controlnet": use_controlnet,
            "lora_enable": lora_enable, "lora_stack": lora_stack,
        })
        paths, info = generate_inpaint(
            prompt, editor_val, resolution, seed,
            negative_prompt=neg, num_steps=steps, guidance_scale=guidance,
            cfg_truncation=cfg_trunc, control_scale=control_scale,
            max_sequence_length=max_seq, time_shift=time_shift,
            lora_stack=lora_stack,
            need_controlnet=use_controlnet,
            progress=progress,
        )
        return paths[0] if paths else None, info

    ip_gen_inpaint.click(
        fn=_do_inpaint,
        inputs=[ip_editor,
                ip_prompt, ip_neg, ip_resolution, ip_seed,
                ip_steps, ip_time_shift, ip_control_scale, ip_guidance, ip_cfg_trunc, ip_max_seq,
                ip_use_controlnet, ip_lora_enable, ip_lora_count,
                *ip_lora_dropdowns, *ip_lora_scales],
        outputs=[ip_result, ip_info],
        concurrency_limit=1,
    )

    # --- Generate outpaint ---
    def _do_outpaint(out_image, direction, expand_px,
                     prompt, neg, resolution, seed,
                     steps, time_shift, control_scale, guidance, cfg_trunc, max_seq,
                     use_controlnet, lora_enable, lora_count,
                     *lora_args,
                     progress=gr.Progress(track_tqdm=True)):
        lora_dds = list(lora_args[:MAX_LORA_STACK])
        lora_sls = list(lora_args[MAX_LORA_STACK:])
        lora_stack = _ip_build_lora_stack(lora_enable, lora_count, *lora_dds, *lora_sls)
        save_gen_ui_params({
            "tab": "inpaint",
            "prompt": prompt, "neg": neg, "resolution": resolution,
            "seed": seed,
            "steps": steps, "time_shift": time_shift,
            "control_scale": control_scale, "guidance": guidance,
            "cfg_trunc": cfg_trunc, "max_seq": max_seq,
            "use_controlnet": use_controlnet,
            "lora_enable": lora_enable, "lora_stack": lora_stack,
        })
        paths, info = generate_outpaint(
            prompt, out_image, direction, expand_px, resolution, seed,
            negative_prompt=neg, num_steps=steps, guidance_scale=guidance,
            cfg_truncation=cfg_trunc, control_scale=control_scale,
            max_sequence_length=max_seq, time_shift=time_shift,
            lora_stack=lora_stack,
            need_controlnet=use_controlnet,
            progress=progress,
        )
        return paths[0] if paths else None, info

    ip_gen_outpaint.click(
        fn=_do_outpaint,
        inputs=[ip_out_image, ip_direction, ip_expand,
                ip_prompt, ip_neg, ip_resolution, ip_seed,
                ip_steps, ip_time_shift, ip_control_scale, ip_guidance, ip_cfg_trunc, ip_max_seq,
                ip_use_controlnet, ip_lora_enable, ip_lora_count,
                *ip_lora_dropdowns, *ip_lora_scales],
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
        "lora_enable": ip_lora_enable,
        "lora_dropdowns": ip_lora_dropdowns, "lora_scales": ip_lora_scales,
        "result": ip_result,
    }
