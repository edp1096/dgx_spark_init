"""LTX-2 Gradio Web UI"""

import argparse
import atexit
import logging
import time
from pathlib import Path

import gradio as gr

from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

from generators import (
    generate_a2vid,
    generate_distilled,
    generate_iclora,
    generate_keyframe,
    generate_retake,
    generate_ti2vid,
    consume_last_gen_result,
    get_active_gen_inputs,
    get_gen_info_for_tab,
    get_last_gen_result,
    get_loading_status,
    get_result_version,
    get_worker_mgr,
    is_generation_active,
    set_model_dir,
)
from i18n import LANGUAGES, STRINGS, get_i18n_js
from pipeline_manager import (
    DEFAULTS,
    IC_LORA_MAP,
    OUTPUT_DIR,
    RESOLUTION_CHOICES,
    SAMPLE_PROMPTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ltx2-ui")


# ---------------------------------------------------------------------------
# Frames / Duration helpers
# ---------------------------------------------------------------------------
def _frames_to_duration(frames, fps):
    """Convert 8k+1 frame count -> duration in seconds."""
    if fps <= 0:
        return 0.0
    return round((int(frames) - 1) / fps, 1)


def _duration_to_frames(duration, fps):
    """Convert seconds -> nearest valid 8k+1 frame count (clamped to 9..257)."""
    raw = duration * fps
    frames = round(raw / 8) * 8 + 1
    return max(9, frames)


def _switch_frame_mode(mode, frames, fps):
    """Toggle visibility and sync values when switching modes."""
    is_frames = mode == "Frames"
    dur = _frames_to_duration(frames, fps)
    return gr.update(visible=is_frames), gr.update(visible=not is_frames, value=dur)


# ---------------------------------------------------------------------------
# Kill / Stop handlers
# ---------------------------------------------------------------------------
def _do_kill():
    """Hard-kill the worker process."""
    mgr = get_worker_mgr()
    msg = mgr.kill()
    logger.info("Kill: %s", msg)
    return msg


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Prompt Constructor
# ---------------------------------------------------------------------------
_PROMPT_STYLES = [
    "", "Cinematic", "Photorealistic", "3D Animation", "Anime",
    "Vintage Film (VHS)", "Film Noir", "Cyberpunk", "Oil Painting", "Claymation",
]
_PROMPT_SHOTS = [
    "", "Wide establishing shot", "Medium shot", "Close-up", "Extreme close-up",
    "Over-the-shoulder", "Low angle", "High angle", "Overhead",
]
_PROMPT_LIGHTING = [
    "", "Natural sunlight", "Golden hour", "Cinematic lighting", "Volumetric fog",
    "Neon glow", "Dark and moody", "Studio lighting", "Soft rim light",
]
_PROMPT_CAMERA = [
    "", "static frame", "handheld movement", "pushes in", "pulls back",
    "tilts upward", "circles around", "pans across", "follows the subject", "tracks sideways",
]


def _build_prompt_text(style, shot, subject, environment, lighting, camera):
    """Build a descriptive prompt sentence from constructor fields."""
    parts = []
    if style:
        parts.append(f"A {style.lower()}")
    else:
        parts.append("A")
    if shot:
        parts.append(f"{shot.lower()} of")
    else:
        parts.append("shot of")
    parts.append(subject.strip() if subject.strip() else "the scene")
    if environment.strip():
        parts.append(f"in {environment.strip()}")
    if lighting:
        parts.append(f"with {lighting.lower()}")
    sentence = " ".join(parts) + "."
    if camera:
        sentence += f" The camera {camera}."
    return sentence


def create_prompt_constructor(prompt_component):
    """Create Prompt Constructor accordion and wire the Insert button."""
    with gr.Accordion("Prompt Constructor", open=False):
        with gr.Row():
            pc_style = gr.Dropdown(_PROMPT_STYLES, value="", label="Style", allow_custom_value=True)
            pc_shot = gr.Dropdown(_PROMPT_SHOTS, value="", label="Shot Type", allow_custom_value=True)
        with gr.Row():
            pc_subject = gr.Textbox(label="Subject & Action", placeholder="a woman walking through a forest")
            pc_env = gr.Textbox(label="Environment", placeholder="a misty mountain trail at dawn")
        with gr.Row():
            pc_lighting = gr.Dropdown(_PROMPT_LIGHTING, value="", label="Lighting", allow_custom_value=True)
            pc_camera = gr.Dropdown(_PROMPT_CAMERA, value="", label="Camera Movement", allow_custom_value=True)
        pc_insert = gr.Button("Insert into Prompt", size="sm", variant="secondary")

    def _insert(style, shot, subject, env, lighting, camera, current_prompt):
        built = _build_prompt_text(style, shot, subject, env, lighting, camera)
        if current_prompt.strip():
            return current_prompt.rstrip() + " " + built
        return built

    pc_insert.click(
        fn=_insert,
        inputs=[pc_style, pc_shot, pc_subject, pc_env, pc_lighting, pc_camera, prompt_component],
        outputs=[prompt_component],
    )


def create_frame_controls(default_frames=121, default_fps=25):
    """Create frame/duration toggle with synced inputs. Returns (frame_mode, frames, duration)."""
    with gr.Row():
        frame_mode = gr.Radio(["Frames", "Duration (sec)"], value="Frames", label="Length", scale=1)
        frames = gr.Number(value=default_frames, label="Frames (8k+1)", precision=0, minimum=9, visible=True, scale=1)
        duration = gr.Number(
            value=_frames_to_duration(default_frames, default_fps),
            label="Duration (sec)", minimum=0.1, step=0.1, visible=False, scale=1,
        )
    return frame_mode, frames, duration


def wire_frame_sync(frame_mode, frames, duration, fps):
    """Wire up sync: duration->frames, fps->frames. Call after fps is created."""
    frame_mode.change(
        fn=_switch_frame_mode,
        inputs=[frame_mode, frames, fps],
        outputs=[frames, duration],
    )
    duration.change(
        fn=lambda d, f: _duration_to_frames(d, f),
        inputs=[duration, fps], outputs=[frames],
    )
    fps.change(
        fn=lambda mode, d, f: _duration_to_frames(d, f) if mode == "Duration (sec)" else gr.update(),
        inputs=[frame_mode, duration, fps], outputs=[frames],
    )


def create_guidance_accordion(
    prefix: str,
    v_defaults=None,
    a_defaults=None,
    show_audio: bool = True,
) -> list:
    """Create guidance parameter controls. Returns list of components."""
    if v_defaults is None:
        v_defaults = DEFAULTS.video_guider_params
    if a_defaults is None:
        a_defaults = DEFAULTS.audio_guider_params

    components = []
    with gr.Accordion("Guidance (advanced)", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Video Guidance**")
                v_cfg = gr.Slider(1.0, 10.0, value=v_defaults.cfg_scale, step=0.1, label="CFG Scale")
                v_stg = gr.Slider(0.0, 3.0, value=v_defaults.stg_scale, step=0.1, label="STG Scale")
                v_rescale = gr.Slider(0.0, 1.0, value=v_defaults.rescale_scale, step=0.05, label="Rescale")
                v_modality = gr.Slider(1.0, 10.0, value=v_defaults.modality_scale, step=0.1, label="Modality Scale")
                v_stg_blocks = gr.Textbox(value=",".join(str(b) for b in v_defaults.stg_blocks), label="STG Blocks")
                v_skip_step = gr.Slider(0, 5, value=0, step=1, label="Skip Step",
                                        info="Skip STG guidance every N steps (0=none, higher=faster)")
                components.extend([v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks, v_skip_step])
            if show_audio:
                with gr.Column():
                    gr.Markdown("**Audio Guidance**")
                    a_cfg = gr.Slider(1.0, 10.0, value=a_defaults.cfg_scale, step=0.1, label="CFG Scale")
                    a_stg = gr.Slider(0.0, 3.0, value=a_defaults.stg_scale, step=0.1, label="STG Scale")
                    a_rescale = gr.Slider(0.0, 1.0, value=a_defaults.rescale_scale, step=0.05, label="Rescale")
                    a_modality = gr.Slider(1.0, 10.0, value=a_defaults.modality_scale, step=0.1, label="Modality Scale")
                    a_stg_blocks = gr.Textbox(value=",".join(str(b) for b in a_defaults.stg_blocks), label="STG Blocks")
                    a_skip_step = gr.Slider(0, 5, value=0, step=1, label="Skip Step",
                                            info="Skip STG guidance every N steps (0=none, higher=faster)")
                    components.extend([a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks, a_skip_step])
    return components


_GUIDANCE_NAMES_VA = [
    "v_cfg", "v_stg", "v_rescale", "v_modality", "v_stg_blocks", "v_skip_step",
    "a_cfg", "a_stg", "a_rescale", "a_modality", "a_stg_blocks", "a_skip_step",
]
_GUIDANCE_NAMES_V = _GUIDANCE_NAMES_VA[:6]


def _i18n_msg(key, lang="en", **kwargs):
    """Return translated string for the given language."""
    texts = STRINGS.get(key, {})
    msg = texts.get(lang, texts.get("en", key))
    if kwargs:
        msg = msg.format(**kwargs)
    return msg


def create_preset_row(tab_key, param_pairs):
    """Create per-tab preset import/export controls.

    param_pairs: list of (param_name, gr_component).
    """
    import json as _pjson

    comps = [c for _, c in param_pairs]

    with gr.Accordion("Presets", open=False):
        with gr.Row():
            p_export = gr.DownloadButton("Export", variant="secondary", size="sm")
            p_import = gr.File(label="Import (.json)", file_types=[".json"], type="filepath")
        p_status = gr.Textbox(label="", interactive=False, visible=False, max_lines=1)
        p_lang = gr.Textbox(value="en", visible=False, elem_id=f"preset-lang-{tab_key}")

    def _do_export(*values):
        import tempfile as _tf
        data = {"tab": tab_key}
        for (name, _), val in zip(param_pairs, values):
            data[name] = val
        path = Path(_tf.mkdtemp()) / f"ltx2_{tab_key}.json"
        path.write_text(_pjson.dumps(data, indent=2, ensure_ascii=False))
        return str(path)

    def _do_import(file_path, lang):
        if file_path is None:
            return [gr.update(visible=False)] + [gr.update()] * len(param_pairs)
        lang = lang or "en"
        try:
            data = _pjson.loads(Path(file_path).read_text())
            file_tab = data.get("tab", "")
            if file_tab != tab_key:
                gr.Warning(_i18n_msg("preset_tab_mismatch", lang=lang, tab=file_tab))
                return [gr.update(visible=True, value=_i18n_msg("preset_tab_mismatch_short", lang=lang, tab=file_tab))] + [gr.update()] * len(param_pairs)
            result = []
            for name, _ in param_pairs:
                result.append(data[name] if name in data else gr.update())
            return [gr.update(visible=True, value=_i18n_msg("preset_loaded", lang=lang))] + result
        except Exception as e:
            return [gr.update(visible=True, value=f"Error: {e}")] + [gr.update()] * len(param_pairs)

    p_export.click(fn=_do_export, inputs=comps, outputs=[p_export])
    p_import.change(
        fn=_do_import, inputs=[p_import, p_lang], outputs=[p_status] + comps,
        js=f"(file, lang) => [file, localStorage.getItem('ltx2-lang') || 'en']",
    )


def create_output_column(gen_type: str):
    """Create standardised output column with video, info, enhanced prompt, kill button, and status."""
    video = gr.Video(label="Generated Video")
    info = gr.Textbox(
        label="Info", interactive=False,
        value=lambda: get_gen_info_for_tab(gen_type), every=2,
    )
    enhanced = gr.Textbox(
        label="Enhanced Prompt", interactive=False, lines=4,
        placeholder="보정된 프롬프트가 여기에 표시됩니다 (Enhance Prompt 활성화 시)",
    )
    with gr.Row():
        kill_btn = gr.Button("Kill (emergency stop)", variant="stop", size="sm", elem_classes=["kill-btn"])
    kill_msg = gr.Textbox(label="", interactive=False, visible=False)
    status_md = gr.Markdown(value=get_loading_status, every=1)

    kill_btn.click(fn=_do_kill, outputs=[kill_msg])

    return video, info, enhanced


# ---------------------------------------------------------------------------
# UI Builder
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    def get_memory_status():
        import psutil
        vm = psutil.virtual_memory()
        used = vm.used / 1024**3
        total = vm.total / 1024**3
        return f"Memory: **{used:.1f}GB/{total:.0f}GB** ({vm.percent:.0f}% used)"

    with gr.Blocks(
        title="LTX-2 Video Generator",
        css=".memory-status { text-align: right; } .kill-btn { background: #dc2626 !important; color: white !important; border: none !important; } .kill-btn:hover { background: #b91c1c !important; }",
        js=get_i18n_js(),
    ) as app:
        with gr.Row():
            gr.Markdown("# LTX-2 Video Generator")
            gr.Markdown(value=get_memory_status, every=3, elem_classes=["memory-status"])

        with gr.Tabs():
            # ==============================================================
            # Tab 1: ti2vid (distilled) — Fast 8-step
            # ==============================================================
            with gr.Tab("ti2vid (distilled)"):
                gr.Markdown("*Fast 8-step generation. Negative prompt via NAG guidance (2x slower when enabled).*")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            t2_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your video...")
                            with gr.Row():
                                t2_sample_btns = []
                                for i in range(len(SAMPLE_PROMPTS)):
                                    t2_sample_btns.append(gr.Button(f"Sample {i+1}", size="sm", min_width=60))
                        create_prompt_constructor(t2_prompt)
                        with gr.Accordion("Negative Prompt (NAG)", open=False):
                            t2_neg = gr.Textbox(label="Negative Prompt", value="", lines=2, show_label=False)
                            t2_nag_scale = gr.Slider(1.0, 15.0, value=1.0, step=0.5, label="NAG Scale",
                                                     info="Guidance strength (1.0=off, higher=stronger, doubles inference time)")
                        with gr.Accordion("Conditioning Image", open=False):
                            t2_image = gr.Image(label="Image (optional)", type="numpy")
                            t2_img_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Image Strength")
                            t2_img_crf = gr.Slider(0, 51, value=33, step=1, label="Image CRF",
                                                   info="Compression quality (0=lossless, 51=worst)")
                        t2_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="768x512", label="Resolution (WxH)", allow_custom_value=True)
                        t2_frame_mode, t2_frames, t2_duration = create_frame_controls()
                        with gr.Row():
                            t2_fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
                            t2_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        with gr.Row():
                            t2_enhance = gr.Checkbox(value=False, label="Enhance Prompt")
                            t2_fp8 = gr.Checkbox(value=True, label="FP8 Quantization", interactive=False)
                            t2_no_audio = gr.Checkbox(value=False, label="Disable Audio")
                        create_preset_row("distilled", [
                            ("prompt", t2_prompt), ("negative_prompt", t2_neg),
                            ("nag_scale", t2_nag_scale),
                            ("img_strength", t2_img_strength), ("img_crf", t2_img_crf),
                            ("resolution", t2_resolution), ("frames", t2_frames),
                            ("fps", t2_fps), ("seed", t2_seed),
                            ("enhance", t2_enhance), ("no_audio", t2_no_audio),
                            ("frame_mode", t2_frame_mode), ("duration", t2_duration),
                        ])
                        t2_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t2_video, t2_info, t2_enhanced = create_output_column("distilled")

                wire_frame_sync(t2_frame_mode, t2_frames, t2_duration, t2_fps)

                t2_btn.click(
                    fn=lambda: (None, "", ""),
                    outputs=[t2_video, t2_info, t2_enhanced],
                ).then(
                    fn=generate_distilled,
                    inputs=[
                        t2_prompt, t2_neg, t2_nag_scale,
                        t2_image, t2_img_strength, t2_img_crf,
                        t2_resolution, t2_frames, t2_fps, t2_seed,
                        t2_enhance, t2_fp8,
                        t2_frame_mode, t2_duration, t2_no_audio,
                    ],
                    outputs=[t2_video, t2_info, t2_enhanced],
                )
                for i, btn in enumerate(t2_sample_btns):
                    btn.click(fn=lambda idx=i: SAMPLE_PROMPTS[idx], outputs=[t2_prompt])

            # ==============================================================
            # Tab 2: ti2vid — Full dev model
            # ==============================================================
            with gr.Tab("ti2vid"):
                gr.Markdown("*2-stage generation (dev model). Supports negative prompt and guidance parameters.*")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            t1_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your video...")
                            with gr.Row():
                                t1_sample_btns = []
                                for i in range(len(SAMPLE_PROMPTS)):
                                    t1_sample_btns.append(gr.Button(f"Sample {i+1}", size="sm", min_width=60))
                        create_prompt_constructor(t1_prompt)
                        with gr.Accordion("Negative Prompt", open=False):
                            t1_neg = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2, show_label=False)
                        with gr.Accordion("Conditioning Image", open=False):
                            t1_image = gr.Image(label="Image (optional)", type="numpy")
                            t1_img_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Image Strength")
                            t1_img_crf = gr.Slider(0, 51, value=33, step=1, label="Image CRF",
                                                   info="Compression quality (0=lossless, 51=worst)")
                        t1_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="1024x768", label="Resolution (WxH)", allow_custom_value=True)
                        t1_frame_mode, t1_frames, t1_duration = create_frame_controls()
                        with gr.Row():
                            t1_fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
                            t1_steps = gr.Slider(1, 50, value=DEFAULTS.num_inference_steps, step=1, label="Steps")
                        with gr.Row():
                            t1_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                            t1_sampler = gr.Radio(["euler", "res_2s"], value="euler", label="Sampler")
                        with gr.Row():
                            t1_enhance = gr.Checkbox(value=False, label="Enhance Prompt")
                            t1_fp8 = gr.Checkbox(value=True, label="FP8 Quantization", interactive=False)
                            t1_no_audio = gr.Checkbox(value=False, label="Disable Audio")
                        t1_guidance = create_guidance_accordion("t1")
                        create_preset_row("ti2vid", [
                            ("prompt", t1_prompt), ("negative_prompt", t1_neg),
                            ("img_strength", t1_img_strength), ("img_crf", t1_img_crf),
                            ("resolution", t1_resolution), ("frames", t1_frames),
                            ("fps", t1_fps), ("steps", t1_steps),
                            ("seed", t1_seed), ("sampler", t1_sampler),
                            ("enhance", t1_enhance), ("no_audio", t1_no_audio),
                            ("frame_mode", t1_frame_mode), ("duration", t1_duration),
                        ] + [(_n, t1_guidance[_i]) for _i, _n in enumerate(_GUIDANCE_NAMES_VA)])
                        t1_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t1_video, t1_info, t1_enhanced = create_output_column("ti2vid")

                wire_frame_sync(t1_frame_mode, t1_frames, t1_duration, t1_fps)

                t1_btn.click(
                    fn=lambda: (None, "", ""),
                    outputs=[t1_video, t1_info, t1_enhanced],
                ).then(
                    fn=generate_ti2vid,
                    inputs=[
                        t1_prompt, t1_neg, t1_image, t1_img_strength, t1_img_crf,
                        t1_resolution, t1_frames, t1_fps, t1_steps, t1_seed, t1_sampler,
                        t1_enhance, t1_fp8,
                        *t1_guidance,
                        t1_frame_mode, t1_duration, t1_no_audio,
                    ],
                    outputs=[t1_video, t1_info, t1_enhanced],
                )
                for i, btn in enumerate(t1_sample_btns):
                    btn.click(fn=lambda idx=i: SAMPLE_PROMPTS[idx], outputs=[t1_prompt])

            # ==============================================================
            # Tab 3: IC-LoRA
            # ==============================================================
            with gr.Tab("IC-LoRA"):
                gr.Markdown("*Reference video conditioned generation (distilled model). Negative prompt via NAG guidance.*")
                with gr.Row():
                    with gr.Column(scale=1):
                        t3_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the transformation...")
                        create_prompt_constructor(t3_prompt)
                        with gr.Accordion("Negative Prompt (NAG)", open=False):
                            t3_neg = gr.Textbox(label="Negative Prompt", value="", lines=2, show_label=False)
                            t3_nag_scale = gr.Slider(1.0, 15.0, value=1.0, step=0.5, label="NAG Scale",
                                                     info="Guidance strength (1.0=off, higher=stronger, doubles inference time)")
                        t3_ref_video = gr.Video(label="Reference Video", sources=["upload"])
                        t3_ref_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Reference Strength")
                        t3_lora = gr.Dropdown(
                            list(IC_LORA_MAP.keys()),
                            value="Union Control", label="IC-LoRA Type",
                            info="Union Control: Preserve overall structure of reference video | Motion Track: Follow motion trajectory",
                        )
                        t3_attn_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Attention Strength")
                        with gr.Accordion("Conditioning Image", open=False):
                            t3_image = gr.Image(label="Image (optional)", type="numpy")
                            t3_img_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Image Strength")
                            t3_img_crf = gr.Slider(0, 51, value=33, step=1, label="Image CRF",
                                                   info="Compression quality (0=lossless, 51=worst)")
                        t3_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="768x512", label="Resolution (WxH)", allow_custom_value=True)
                        t3_frame_mode, t3_frames, t3_duration = create_frame_controls()
                        with gr.Row():
                            t3_fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
                            t3_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        with gr.Row():
                            t3_skip_stage2 = gr.Checkbox(value=False, label="Skip Upscale (half res)")
                            t3_enhance = gr.Checkbox(value=False, label="Enhance Prompt")
                            t3_fp8 = gr.Checkbox(value=True, label="FP8 Quantization", interactive=False)
                            t3_no_audio = gr.Checkbox(value=False, label="Disable Audio")
                        create_preset_row("iclora", [
                            ("prompt", t3_prompt), ("negative_prompt", t3_neg),
                            ("nag_scale", t3_nag_scale),
                            ("ref_strength", t3_ref_strength), ("lora_type", t3_lora),
                            ("attn_strength", t3_attn_strength),
                            ("img_strength", t3_img_strength), ("img_crf", t3_img_crf),
                            ("resolution", t3_resolution), ("frames", t3_frames),
                            ("fps", t3_fps), ("seed", t3_seed),
                            ("skip_stage2", t3_skip_stage2),
                            ("enhance", t3_enhance), ("no_audio", t3_no_audio),
                            ("frame_mode", t3_frame_mode), ("duration", t3_duration),
                        ])
                        t3_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t3_video, t3_info, t3_enhanced = create_output_column("iclora")

                wire_frame_sync(t3_frame_mode, t3_frames, t3_duration, t3_fps)

                t3_btn.click(
                    fn=lambda: (None, "", ""),
                    outputs=[t3_video, t3_info, t3_enhanced],
                ).then(
                    fn=generate_iclora,
                    inputs=[
                        t3_prompt, t3_neg, t3_nag_scale,
                        t3_ref_video, t3_ref_strength, t3_lora, t3_attn_strength,
                        t3_image, t3_img_strength, t3_img_crf,
                        t3_resolution, t3_frames, t3_fps, t3_seed,
                        t3_skip_stage2, t3_enhance, t3_fp8,
                        t3_frame_mode, t3_duration, t3_no_audio,
                    ],
                    outputs=[t3_video, t3_info, t3_enhanced],
                )

            # ==============================================================
            # Tab 4: Keyframe Interpolation
            # ==============================================================
            with gr.Tab("Keyframe Interpolation"):
                gr.Markdown("*Interpolate between keyframe images (dev model). Supports negative prompt.*")
                with gr.Row():
                    with gr.Column(scale=1):
                        t4_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the interpolation...")
                        create_prompt_constructor(t4_prompt)
                        with gr.Accordion("Negative Prompt", open=False):
                            t4_neg = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2, show_label=False)
                        t4_keyframes = gr.File(label="Keyframe Images", file_count="multiple", file_types=["image"])
                        t4_indices = gr.Textbox(label="Frame Indices (comma-separated)", value="0,120", placeholder="0,60,120")
                        t4_img_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Keyframe Strength")
                        t4_img_crf = gr.Slider(0, 51, value=33, step=1, label="Image CRF",
                                               info="Compression quality (0=lossless, 51=worst)")
                        t4_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="1024x768", label="Resolution (WxH)", allow_custom_value=True)
                        t4_frame_mode, t4_frames, t4_duration = create_frame_controls()
                        with gr.Row():
                            t4_fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
                            t4_steps = gr.Slider(1, 50, value=DEFAULTS.num_inference_steps, step=1, label="Steps")
                        with gr.Row():
                            t4_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        with gr.Row():
                            t4_enhance = gr.Checkbox(value=False, label="Enhance Prompt")
                            t4_fp8 = gr.Checkbox(value=True, label="FP8 Quantization", interactive=False)
                            t4_no_audio = gr.Checkbox(value=False, label="Disable Audio")
                        t4_guidance = create_guidance_accordion("t4")
                        create_preset_row("keyframe", [
                            ("prompt", t4_prompt), ("negative_prompt", t4_neg),
                            ("indices", t4_indices),
                            ("img_strength", t4_img_strength), ("img_crf", t4_img_crf),
                            ("resolution", t4_resolution), ("frames", t4_frames),
                            ("fps", t4_fps), ("steps", t4_steps), ("seed", t4_seed),
                            ("enhance", t4_enhance), ("no_audio", t4_no_audio),
                            ("frame_mode", t4_frame_mode), ("duration", t4_duration),
                        ] + [(_n, t4_guidance[_i]) for _i, _n in enumerate(_GUIDANCE_NAMES_VA)])
                        t4_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t4_video, t4_info, t4_enhanced = create_output_column("keyframe")

                wire_frame_sync(t4_frame_mode, t4_frames, t4_duration, t4_fps)

                t4_btn.click(
                    fn=lambda: (None, "", ""),
                    outputs=[t4_video, t4_info, t4_enhanced],
                ).then(
                    fn=generate_keyframe,
                    inputs=[
                        t4_prompt, t4_neg,
                        t4_keyframes, t4_indices, t4_img_strength, t4_img_crf,
                        t4_resolution, t4_frames, t4_fps, t4_steps, t4_seed,
                        t4_enhance, t4_fp8,
                        *t4_guidance,
                        t4_frame_mode, t4_duration, t4_no_audio,
                    ],
                    outputs=[t4_video, t4_info, t4_enhanced],
                )
            # ==============================================================
            # Tab 5: Audio -> Video
            # ==============================================================
            with gr.Tab("Audio -> Video"):
                gr.Markdown("*Audio-conditioned video generation (dev model). Supports negative prompt.*")
                with gr.Row():
                    with gr.Column(scale=1):
                        t5_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the video for this audio...")
                        create_prompt_constructor(t5_prompt)
                        with gr.Accordion("Negative Prompt", open=False):
                            t5_neg = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2, show_label=False)
                        t5_audio = gr.Audio(label="Audio File", type="filepath")
                        with gr.Row():
                            t5_audio_start = gr.Number(value=0.0, label="Audio Start (sec)")
                            t5_audio_max = gr.Number(value=0.0, label="Max Duration (0=all)")
                        with gr.Accordion("Conditioning Image", open=False):
                            t5_image = gr.Image(label="Image (optional)", type="numpy")
                            t5_img_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Image Strength")
                            t5_img_crf = gr.Slider(0, 51, value=33, step=1, label="Image CRF",
                                                   info="Compression quality (0=lossless, 51=worst)")
                        t5_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="1024x768", label="Resolution (WxH)", allow_custom_value=True)
                        t5_frame_mode, t5_frames, t5_duration = create_frame_controls()
                        with gr.Row():
                            t5_fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
                            t5_steps = gr.Slider(1, 50, value=DEFAULTS.num_inference_steps, step=1, label="Steps")
                        with gr.Row():
                            t5_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        with gr.Row():
                            t5_enhance = gr.Checkbox(value=False, label="Enhance Prompt")
                            t5_fp8 = gr.Checkbox(value=True, label="FP8 Quantization", interactive=False)
                        t5_guidance = create_guidance_accordion("t5", show_audio=False)
                        create_preset_row("a2vid", [
                            ("prompt", t5_prompt), ("negative_prompt", t5_neg),
                            ("audio_start", t5_audio_start), ("audio_max", t5_audio_max),
                            ("img_strength", t5_img_strength), ("img_crf", t5_img_crf),
                            ("resolution", t5_resolution), ("frames", t5_frames),
                            ("fps", t5_fps), ("steps", t5_steps), ("seed", t5_seed),
                            ("enhance", t5_enhance),
                            ("frame_mode", t5_frame_mode), ("duration", t5_duration),
                        ] + [(_n, t5_guidance[_i]) for _i, _n in enumerate(_GUIDANCE_NAMES_V)])
                        t5_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t5_video, t5_info, t5_enhanced = create_output_column("a2vid")

                wire_frame_sync(t5_frame_mode, t5_frames, t5_duration, t5_fps)

                t5_btn.click(
                    fn=lambda: (None, "", ""),
                    outputs=[t5_video, t5_info, t5_enhanced],
                ).then(
                    fn=generate_a2vid,
                    inputs=[
                        t5_prompt, t5_neg,
                        t5_audio, t5_audio_start, t5_audio_max,
                        t5_image, t5_img_strength, t5_img_crf,
                        t5_resolution, t5_frames, t5_fps, t5_steps, t5_seed,
                        t5_enhance, t5_fp8,
                        *t5_guidance,
                        t5_frame_mode, t5_duration,
                    ],
                    outputs=[t5_video, t5_info, t5_enhanced],
                )
            # ==============================================================
            # Tab 6: Retake
            # ==============================================================
            with gr.Tab("Retake"):
                gr.Markdown("*Re-generate sections of existing video. Distilled mode disables negative prompt.*")
                with gr.Row():
                    with gr.Column(scale=1):
                        t6_video_in = gr.Video(label="Source Video", sources=["upload"])
                        t6_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the regenerated section...")
                        with gr.Accordion("Negative Prompt", open=False):
                            t6_neg = gr.Textbox(label="Negative Prompt", value="", lines=2, show_label=False)
                            t6_nag_scale = gr.Slider(1.0, 15.0, value=1.0, step=0.5, label="NAG Scale",
                                                     info="Guidance for distilled mode (1.0=off, ignored in full mode)",
                                                     visible=True)
                        with gr.Row():
                            t6_start = gr.Number(value=0.0, label="Start Time (sec)")
                            t6_end = gr.Number(value=2.0, label="End Time (sec)")
                        with gr.Row():
                            t6_regen_video = gr.Checkbox(value=True, label="Regenerate Video")
                            t6_regen_audio = gr.Checkbox(value=True, label="Regenerate Audio")
                        with gr.Row():
                            t6_steps = gr.Slider(1, 50, value=40, step=1, label="Steps")
                            t6_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        with gr.Row():
                            t6_distilled = gr.Checkbox(value=False, label="Distilled Mode")
                            t6_enhance = gr.Checkbox(value=False, label="Enhance Prompt")
                            t6_fp8 = gr.Checkbox(value=True, label="FP8 Quantization", interactive=False)
                        t6_guidance = create_guidance_accordion("t6")
                        create_preset_row("retake", [
                            ("prompt", t6_prompt), ("negative_prompt", t6_neg),
                            ("nag_scale", t6_nag_scale),
                            ("start_time", t6_start), ("end_time", t6_end),
                            ("regen_video", t6_regen_video), ("regen_audio", t6_regen_audio),
                            ("steps", t6_steps), ("seed", t6_seed),
                            ("distilled", t6_distilled), ("enhance", t6_enhance),
                        ] + [(_n, t6_guidance[_i]) for _i, _n in enumerate(_GUIDANCE_NAMES_VA)])
                        t6_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t6_video_out, t6_info, t6_enhanced = create_output_column("retake")

                t6_btn.click(
                    fn=lambda: (None, "", ""),
                    outputs=[t6_video_out, t6_info, t6_enhanced],
                ).then(
                    fn=generate_retake,
                    inputs=[
                        t6_video_in, t6_prompt, t6_neg, t6_nag_scale,
                        t6_start, t6_end,
                        t6_regen_video, t6_regen_audio,
                        t6_steps, t6_seed, t6_distilled,
                        t6_enhance, t6_fp8,
                        *t6_guidance,
                    ],
                    outputs=[t6_video_out, t6_info, t6_enhanced],
                )
            # ==============================================================
            # Settings Tab
            # ==============================================================
            with gr.Tab("Settings"):
                gr.Markdown("## Model Paths")
                from config import MODEL_DIR as _default_model_dir
                s_model_dir = gr.Textbox(value=str(_default_model_dir), label="Model Directory")
                with gr.Row():
                    s_apply = gr.Button("Apply", variant="secondary")
                    s_check = gr.Button("Check Models", variant="secondary")

                s_status = gr.Textbox(label="Status", interactive=False, lines=10)

                def apply_settings(model_dir):
                    set_model_dir(model_dir)
                    return f"Model directory set to: {model_dir}\nWorker will restart on next generation."

                def check_models(model_dir):
                    model_path = Path(model_dir)
                    if not model_path.exists():
                        return f"Directory not found: {model_dir}"

                    all_files = [
                        ("ltx-2.3-22b-dev-fp8.safetensors", "Dev checkpoint (FP8)"),
                        ("ltx-2.3-22b-distilled-fp8.safetensors", "Distilled checkpoint (FP8)"),
                        ("ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "2x Spatial upscaler"),
                        ("ltx-2.3-22b-distilled-lora-384.safetensors", "Distilled LoRA"),
                        ("gemma-3-12b-it-qat-q4_0-unquantized", "Gemma text encoder"),
                    ]
                    ic_loras = [
                        ("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors", "IC-LoRA Union Control"),
                        ("ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors", "IC-LoRA Motion Track"),
                    ]

                    lines = [f"Model directory: {model_dir}\n", "=== Required ==="]
                    for fname, desc in all_files:
                        exists = (model_path / fname).exists()
                        status = "OK" if exists else "MISSING"
                        lines.append(f"  [{status}] {fname} -- {desc}")

                    lines.append("\n=== IC-LoRA (optional) ===")
                    for fname, desc in ic_loras:
                        exists = (model_path / fname).exists()
                        status = "OK" if exists else "---"
                        lines.append(f"  [{status}] {fname}")

                    return "\n".join(lines)

                s_apply.click(fn=apply_settings, inputs=[s_model_dir], outputs=[s_status])
                s_check.click(fn=check_models, inputs=[s_model_dir], outputs=[s_status])

                gr.Markdown("## Language")
                _lang_names = list(LANGUAGES.values())
                _lang_codes = list(LANGUAGES.keys())
                s_lang = gr.Radio(
                    _lang_names, value=_lang_names[0],
                    label="UI Language", elem_id="lang-selector",
                )
                _name_to_code = {v: k for k, v in LANGUAGES.items()}
                import json as _json
                _map_json = _json.dumps(_name_to_code, ensure_ascii=False)
                s_lang.change(fn=lambda x: None, inputs=[s_lang], js=f"""
                    (val) => {{
                        const map = {_map_json};
                        const lang = map[val] || 'en';
                        if (window._ltx2_setLang) window._ltx2_setLang(lang);
                        return val;
                    }}
                """)

                gr.Markdown("## Log")
                from config import LOG_DIR
                LOG_FILE = LOG_DIR / "gradio.log"
                with gr.Row():
                    s_log_toggle = gr.Button("Show Log", variant="secondary")
                    s_log_download = gr.DownloadButton("Download Log", variant="secondary")
                s_log_text = gr.Textbox(label="Log", interactive=False, lines=15, visible=False)
                _log_visible = gr.State(False)

                def toggle_log(visible):
                    if visible:
                        return gr.update(visible=False, value=""), not visible, gr.update(value="Show Log")
                    if not LOG_FILE.exists():
                        return gr.update(visible=True, value="Log file not found."), not visible, gr.update(value="Hide Log")
                    lines = LOG_FILE.read_text().splitlines()
                    return gr.update(visible=True, value="\n".join(lines[-50:])), not visible, gr.update(value="Hide Log")

                def download_log():
                    if LOG_FILE.exists():
                        return str(LOG_FILE)
                    return None

                s_log_toggle.click(fn=toggle_log, inputs=[_log_visible], outputs=[s_log_text, _log_visible, s_log_toggle])
                s_log_download.click(fn=download_log, outputs=[s_log_download])

            # ==============================================================
            # History Tab
            # ==============================================================
            with gr.Tab("History"):
                gr.Markdown("## Generation History")
                gr.Markdown(f"Output directory: `{OUTPUT_DIR}`")

                with gr.Row():
                    h_refresh = gr.Button("Refresh", variant="secondary")
                    h_delete = gr.Button("Delete Selected", variant="stop")
                    h_flush = gr.Button("Delete All", variant="stop")

                h_file_list = gr.Dropdown(label="Generated Videos", choices=[], interactive=True)
                h_video = gr.Video(label="Preview")
                h_file_info = gr.Textbox(label="File Info", interactive=False)

                def list_outputs():
                    files = sorted(Path(OUTPUT_DIR).glob("ltx2_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
                    choices = [f.name for f in files]
                    return gr.update(choices=choices, value=choices[0] if choices else None)

                def preview_file(filename):
                    if not filename:
                        return None, ""
                    path = Path(OUTPUT_DIR) / filename
                    if not path.exists():
                        return None, "File not found."
                    size_mb = path.stat().st_size / 1024 / 1024
                    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
                    return str(path), f"Size: {size_mb:.1f}MB | Created: {mtime}"

                def delete_file(filename):
                    if not filename:
                        return list_outputs(), None, "No file selected."
                    path = Path(OUTPUT_DIR) / filename
                    if path.exists():
                        path.unlink()
                        logger.info("Deleted: %s", filename)
                    return list_outputs(), None, f"Deleted: {filename}"

                def flush_all():
                    files = list(Path(OUTPUT_DIR).glob("ltx2_*.mp4"))
                    count = len(files)
                    for f in files:
                        f.unlink()
                    logger.info("Flushed %d files", count)
                    return list_outputs(), None, f"Deleted {count} files."

                h_refresh.click(fn=list_outputs, outputs=[h_file_list])
                h_file_list.change(fn=preview_file, inputs=[h_file_list], outputs=[h_video, h_file_info])
                h_delete.click(fn=delete_file, inputs=[h_file_list], outputs=[h_file_list, h_video, h_file_info])
                h_flush.click(
                    fn=flush_all, outputs=[h_file_list, h_video, h_file_info],
                    js="(()=>{const msg=window._ltx2_confirm_msg||'Delete ALL generated videos?';if(!confirm(msg))throw new Error('cancelled')})",
                )

                # Auto-refresh on tab load
                app.load(fn=list_outputs, outputs=[h_file_list])

        # ==============================================================
        # Hidden result monitor — reconnects video output after refresh
        # Info text is self-polling via every=2 in create_output_column.
        # ==============================================================
        _all_videos = [t1_video, t2_video, t3_video, t4_video, t5_video, t6_video_out]
        _tab_map = {
            "ti2vid": 0, "distilled": 1, "iclora": 2,
            "keyframe": 3, "a2vid": 4, "retake": 5,
        }

        def _restore_video():
            """Return last video to the correct tab. No-op during generation."""
            if is_generation_active():
                return [gr.update()] * 6
            result = get_last_gen_result()
            if result is None:
                return [gr.update()] * 6
            if result.get("consumed"):
                return [gr.update()] * 6
            if time.time() - result["time"] > 600:
                return [gr.update()] * 6
            path = result.get("path")
            if not path or not Path(path).exists():
                return [gr.update()] * 6
            consume_last_gen_result()
            idx = _tab_map.get(result["gen_type"], 0)
            outputs = [gr.update()] * 6
            outputs[idx] = path
            return outputs

        # Poll result version every 2s — fires .change() when generation completes
        _result_ver_box = gr.Number(
            value=get_result_version, every=2, visible=False,
        )
        _result_ver_box.change(fn=_restore_video, outputs=_all_videos)

        # ==============================================================
        # Restore inputs + video on page load (refresh during generation)
        # ==============================================================
        # Input components per tab, same order as generate_* function args
        _tab_inputs = {
            "ti2vid": [
                t1_prompt, t1_neg, t1_image, t1_img_strength, t1_img_crf,
                t1_resolution, t1_frames, t1_fps, t1_steps, t1_seed, t1_sampler,
                t1_enhance, t1_fp8,
                *t1_guidance,
                t1_frame_mode, t1_duration, t1_no_audio,
            ],
            "distilled": [
                t2_prompt, t2_image, t2_img_strength, t2_img_crf,
                t2_resolution, t2_frames, t2_fps, t2_seed,
                t2_enhance, t2_fp8,
                t2_frame_mode, t2_duration, t2_no_audio,
            ],
            "iclora": [
                t3_prompt, t3_ref_video, t3_ref_strength, t3_lora, t3_attn_strength,
                t3_image, t3_img_strength, t3_img_crf,
                t3_resolution, t3_frames, t3_fps, t3_seed,
                t3_skip_stage2, t3_enhance, t3_fp8,
                t3_frame_mode, t3_duration, t3_no_audio,
            ],
            "keyframe": [
                t4_prompt, t4_neg,
                t4_keyframes, t4_indices, t4_img_strength, t4_img_crf,
                t4_resolution, t4_frames, t4_fps, t4_steps, t4_seed,
                t4_enhance, t4_fp8, *t4_guidance,
                t4_frame_mode, t4_duration, t4_no_audio,
            ],
            "a2vid": [
                t5_prompt, t5_neg,
                t5_audio, t5_audio_start, t5_audio_max,
                t5_image, t5_img_strength, t5_img_crf,
                t5_resolution, t5_frames, t5_fps, t5_steps, t5_seed,
                t5_enhance, t5_fp8, *t5_guidance,
                t5_frame_mode, t5_duration,
            ],
            "retake": [
                t6_video_in, t6_prompt, t6_neg,
                t6_start, t6_end,
                t6_regen_video, t6_regen_audio,
                t6_steps, t6_seed, t6_distilled,
                t6_enhance, t6_fp8, *t6_guidance,
            ],
        }
        # Flatten all inputs across all tabs (fixed order)
        _all_input_components = []
        _tab_input_ranges = {}  # gen_type -> (start, count)
        for gt in ["ti2vid", "distilled", "iclora", "keyframe", "a2vid", "retake"]:
            start = len(_all_input_components)
            _all_input_components.extend(_tab_inputs[gt])
            _tab_input_ranges[gt] = (start, len(_tab_inputs[gt]))

        _all_restore_outputs = _all_videos + _all_input_components

        # For tabs with frame controls: index of (frames_comp, duration_comp) within _tab_inputs
        _frame_vis_indices = {}
        _frame_comp_map = {
            "ti2vid": (t1_frame_mode, t1_frames, t1_duration),
            "distilled": (t2_frame_mode, t2_frames, t2_duration),
            "iclora": (t3_frame_mode, t3_frames, t3_duration),
            "keyframe": (t4_frame_mode, t4_frames, t4_duration),
            "a2vid": (t5_frame_mode, t5_frames, t5_duration),
        }
        for gt, comps in _tab_inputs.items():
            if gt not in _frame_comp_map:
                continue
            fmc, fc, dc = _frame_comp_map[gt]
            mi = fi = di = None
            for j, c in enumerate(comps):
                if c is fmc:
                    mi = j
                if c is fc:
                    fi = j
                if c is dc:
                    di = j
            if mi is not None and fi is not None and di is not None:
                _frame_vis_indices[gt] = (mi, fi, di)

        def _restore_on_load():
            """Restore active tab inputs on page load (no video — that arrives via _result_ver_box)."""
            n_videos = len(_all_videos)
            n_inputs = len(_all_input_components)
            n_total = n_videos + n_inputs
            outputs = [gr.update()] * n_total

            # Restore inputs if generation is active
            snap = get_active_gen_inputs()
            if snap:
                gt = snap["gen_type"]
                values = snap["values"]
                rng = _tab_input_ranges.get(gt)
                if rng and len(values) == rng[1]:
                    start = n_videos + rng[0]
                    for i, val in enumerate(values):
                        outputs[start + i] = val
                    # Fix visibility for frames / duration based on frame_mode
                    if gt in _frame_vis_indices:
                        mi, fi, di = _frame_vis_indices[gt]
                        frame_mode_val = values[mi]
                        is_frames = frame_mode_val == "Frames"
                        outputs[start + fi] = gr.update(value=values[fi], visible=is_frames)
                        outputs[start + di] = gr.update(value=values[di], visible=not is_frames)

            return outputs

        app.load(fn=_restore_on_load, outputs=_all_restore_outputs)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-2 Gradio Web UI")
    parser.add_argument("--server-name", default="0.0.0.0", help="Server hostname")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    from config import MODEL_DIR as _default_model_dir
    parser.add_argument("--model-dir", default=str(_default_model_dir), help="Model directory path")
    args = parser.parse_args()

    # Initialize worker manager with model dir
    mgr = get_worker_mgr()
    mgr.model_dir = args.model_dir

    # Start worker eagerly (so model import happens during startup, not on first generate)
    mgr.ensure_running()

    # Clean shutdown on exit
    atexit.register(mgr.stop)

    app = build_ui()
    app.queue()
    app.launch(server_name=args.server_name, server_port=args.server_port, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
