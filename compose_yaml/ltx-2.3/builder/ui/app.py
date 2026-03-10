"""LTX-2 Gradio Web UI"""

import argparse
import logging
import time
from pathlib import Path

import gradio as gr
import torch

from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

from generators import (
    generate_a2vid,
    generate_distilled,
    generate_iclora,
    generate_keyframe,
    generate_retake,
    generate_ti2vid,
)
from pipeline_manager import (
    DEFAULTS,
    IC_LORA_MAP,
    OUTPUT_DIR,
    RESOLUTION_CHOICES,
    SAMPLE_PROMPTS,
    pipeline_mgr,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ltx2-ui")


# ---------------------------------------------------------------------------
# Frames / Duration helpers
# ---------------------------------------------------------------------------
def _frames_to_duration(frames, fps):
    """Convert 8k+1 frame count → duration in seconds."""
    if fps <= 0:
        return 0.0
    return round((int(frames) - 1) / fps, 1)


def _duration_to_frames(duration, fps):
    """Convert seconds → nearest valid 8k+1 frame count (clamped to 9..257)."""
    raw = duration * fps
    frames = round(raw / 8) * 8 + 1
    return max(9, min(257, frames))


def _switch_frame_mode(mode, frames, fps):
    """Toggle visibility and sync values when switching modes."""
    is_frames = mode == "Frames"
    dur = _frames_to_duration(frames, fps)
    return gr.update(visible=is_frames), gr.update(visible=not is_frames, value=dur)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
def create_frame_controls(default_frames=121, default_fps=25):
    """Create frame/duration toggle with synced inputs. Returns (frame_mode, frames, duration)."""
    with gr.Row():
        frame_mode = gr.Radio(["Frames", "Duration (sec)"], value="Frames", label="Length", scale=1)
        frames = gr.Number(value=default_frames, label="Frames (8k+1)", precision=0, minimum=9, maximum=257, visible=True, scale=1)
        duration = gr.Number(
            value=_frames_to_duration(default_frames, default_fps),
            label="Duration (sec)", minimum=0.1, step=0.1, visible=False, scale=1,
        )
    return frame_mode, frames, duration


def wire_frame_sync(frame_mode, frames, duration, fps):
    """Wire up sync: duration→frames, fps→frames. Call after fps is created."""
    # When switching mode, sync duration from current frames
    frame_mode.change(
        fn=_switch_frame_mode,
        inputs=[frame_mode, frames, fps],
        outputs=[frames, duration],
    )
    # Duration edits → update frames (hidden but used by generator)
    duration.change(
        fn=lambda d, f: _duration_to_frames(d, f),
        inputs=[duration, fps], outputs=[frames],
    )
    # FPS change → recompute frames if in Duration mode
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
                components.extend([v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks])
            if show_audio:
                with gr.Column():
                    gr.Markdown("**Audio Guidance**")
                    a_cfg = gr.Slider(1.0, 10.0, value=a_defaults.cfg_scale, step=0.1, label="CFG Scale")
                    a_stg = gr.Slider(0.0, 3.0, value=a_defaults.stg_scale, step=0.1, label="STG Scale")
                    a_rescale = gr.Slider(0.0, 1.0, value=a_defaults.rescale_scale, step=0.05, label="Rescale")
                    a_modality = gr.Slider(1.0, 10.0, value=a_defaults.modality_scale, step=0.1, label="Modality Scale")
                    a_stg_blocks = gr.Textbox(value=",".join(str(b) for b in a_defaults.stg_blocks), label="STG Blocks")
                    components.extend([a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks])
    return components


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

    with gr.Blocks(title="LTX-2 Video Generator", css=".memory-status { text-align: right; }") as app:
        with gr.Row():
            gr.Markdown("# LTX-2 Video Generator")
            gr.Markdown(value=get_memory_status, every=3, elem_classes=["memory-status"])

        with gr.Tabs():
            # ==============================================================
            # Tab 1: Text/Image -> Video
            # ==============================================================
            with gr.Tab("Text/Image → Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            t1_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your video...")
                            with gr.Row():
                                t1_sample_btns = []
                                for i in range(len(SAMPLE_PROMPTS)):
                                    t1_sample_btns.append(gr.Button(f"Sample {i+1}", size="sm", min_width=60))
                        with gr.Accordion("Negative Prompt", open=False):
                            t1_neg = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2, show_label=False)
                        with gr.Accordion("Conditioning Image", open=False):
                            t1_image = gr.Image(label="Image (optional)", type="numpy")
                            t1_img_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Image Strength")
                        t1_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="768x512", label="Resolution (WxH)", allow_custom_value=True)
                        t1_frame_mode, t1_frames, t1_duration = create_frame_controls()
                        with gr.Row():
                            t1_fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
                            t1_steps = gr.Slider(1, 50, value=DEFAULTS.num_inference_steps, step=1, label="Steps")
                        with gr.Row():
                            t1_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                            t1_sampler = gr.Radio(["euler", "res_2s"], value="euler", label="Sampler")
                        with gr.Row():
                            t1_enhance = gr.Checkbox(value=False, label="Enhance Prompt")
                            t1_fp8 = gr.Checkbox(value=True, label="FP8 Quantization")
                        t1_guidance = create_guidance_accordion("t1")
                        t1_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t1_video = gr.Video(label="Generated Video")
                        t1_info = gr.Textbox(label="Info", interactive=False)
                        gr.Markdown(value=pipeline_mgr.get_loading_status, every=1)

                wire_frame_sync(t1_frame_mode, t1_frames, t1_duration, t1_fps)

                t1_btn.click(
                    fn=lambda: (None, ""),
                    outputs=[t1_video, t1_info],
                ).then(
                    fn=generate_ti2vid,
                    inputs=[
                        t1_prompt, t1_neg, t1_image, t1_img_strength,
                        t1_resolution, t1_frames, t1_fps, t1_steps, t1_seed, t1_sampler,
                        t1_enhance, t1_fp8,
                        *t1_guidance,
                    ],
                    outputs=[t1_video, t1_info],
                )
                for i, btn in enumerate(t1_sample_btns):
                    btn.click(fn=lambda idx=i: SAMPLE_PROMPTS[idx], outputs=[t1_prompt])

            # ==============================================================
            # Tab 2: Distilled (Fast)
            # ==============================================================
            with gr.Tab("Distilled (Fast)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            t2_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your video...")
                            with gr.Row():
                                t2_sample_btns = []
                                for i in range(len(SAMPLE_PROMPTS)):
                                    t2_sample_btns.append(gr.Button(f"Sample {i+1}", size="sm", min_width=60))
                        with gr.Accordion("Conditioning Image", open=False):
                            t2_image = gr.Image(label="Image (optional)", type="numpy")
                            t2_img_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Image Strength")
                        t2_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="768x512", label="Resolution (WxH)", allow_custom_value=True)
                        t2_frame_mode, t2_frames, t2_duration = create_frame_controls()
                        with gr.Row():
                            t2_fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
                            t2_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        with gr.Row():
                            t2_enhance = gr.Checkbox(value=False, label="Enhance Prompt")
                            t2_fp8 = gr.Checkbox(value=True, label="FP8 Quantization")
                        gr.Markdown("*Fixed 8-step distilled schedule. No guidance parameters.*")
                        t2_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t2_video = gr.Video(label="Generated Video")
                        t2_info = gr.Textbox(label="Info", interactive=False)
                        gr.Markdown(value=pipeline_mgr.get_loading_status, every=1)

                wire_frame_sync(t2_frame_mode, t2_frames, t2_duration, t2_fps)

                t2_btn.click(
                    fn=lambda: (None, ""),
                    outputs=[t2_video, t2_info],
                ).then(
                    fn=generate_distilled,
                    inputs=[
                        t2_prompt, t2_image, t2_img_strength,
                        t2_resolution, t2_frames, t2_fps, t2_seed,
                        t2_enhance, t2_fp8,
                    ],
                    outputs=[t2_video, t2_info],
                )
                for i, btn in enumerate(t2_sample_btns):
                    btn.click(fn=lambda idx=i: SAMPLE_PROMPTS[idx], outputs=[t2_prompt])

            # ==============================================================
            # Tab 3: IC-LoRA
            # ==============================================================
            with gr.Tab("IC-LoRA"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t3_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the transformation...")
                        t3_ref_video = gr.Video(label="Reference Video", sources=["upload"])
                        t3_ref_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Reference Strength")
                        t3_lora = gr.Dropdown(
                            list(IC_LORA_MAP.keys()),
                            value="Union Control", label="IC-LoRA Type",
                            info="Union Control: Preserve overall structure of reference video | Inpainting: Regenerate partial regions | Motion Track: Follow motion trajectory | Detailer: Enhance fine details | Pose Control: Control body poses",
                        )
                        t3_attn_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Attention Strength")
                        with gr.Accordion("Conditioning Image", open=False):
                            t3_image = gr.Image(label="Image (optional)", type="numpy")
                            t3_img_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Image Strength")
                        t3_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="768x512", label="Resolution (WxH)", allow_custom_value=True)
                        t3_frame_mode, t3_frames, t3_duration = create_frame_controls()
                        with gr.Row():
                            t3_fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
                            t3_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        with gr.Row():
                            t3_skip_stage2 = gr.Checkbox(value=False, label="Skip Upscale (half res)")
                            t3_enhance = gr.Checkbox(value=False, label="Enhance Prompt")
                            t3_fp8 = gr.Checkbox(value=True, label="FP8 Quantization")
                        t3_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t3_video = gr.Video(label="Generated Video")
                        t3_info = gr.Textbox(label="Info", interactive=False)
                        gr.Markdown(value=pipeline_mgr.get_loading_status, every=1)


                wire_frame_sync(t3_frame_mode, t3_frames, t3_duration, t3_fps)

                t3_btn.click(
                    fn=lambda: (None, ""),
                    outputs=[t3_video, t3_info],
                ).then(
                    fn=generate_iclora,
                    inputs=[
                        t3_prompt, t3_ref_video, t3_ref_strength, t3_lora, t3_attn_strength,
                        t3_image, t3_img_strength,
                        t3_resolution, t3_frames, t3_fps, t3_seed,
                        t3_skip_stage2, t3_enhance, t3_fp8,
                    ],
                    outputs=[t3_video, t3_info],
                )

            # ==============================================================
            # Tab 4: Keyframe Interpolation
            # ==============================================================
            with gr.Tab("Keyframe Interpolation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t4_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the interpolation...")
                        with gr.Accordion("Negative Prompt", open=False):
                            t4_neg = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2, show_label=False)
                        t4_keyframes = gr.File(label="Keyframe Images", file_count="multiple", file_types=["image"])
                        t4_indices = gr.Textbox(label="Frame Indices (comma-separated)", value="0,120", placeholder="0,60,120")
                        t4_img_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Keyframe Strength")
                        t4_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="768x512", label="Resolution (WxH)", allow_custom_value=True)
                        t4_frame_mode, t4_frames, t4_duration = create_frame_controls()
                        with gr.Row():
                            t4_fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
                            t4_steps = gr.Slider(1, 50, value=DEFAULTS.num_inference_steps, step=1, label="Steps")
                        with gr.Row():
                            t4_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        with gr.Row():
                            t4_enhance = gr.Checkbox(value=False, label="Enhance Prompt")
                            t4_fp8 = gr.Checkbox(value=True, label="FP8 Quantization")
                        t4_guidance = create_guidance_accordion("t4")
                        t4_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t4_video = gr.Video(label="Generated Video")
                        t4_info = gr.Textbox(label="Info", interactive=False)
                        gr.Markdown(value=pipeline_mgr.get_loading_status, every=1)

                wire_frame_sync(t4_frame_mode, t4_frames, t4_duration, t4_fps)

                t4_btn.click(
                    fn=lambda: (None, ""),
                    outputs=[t4_video, t4_info],
                ).then(
                    fn=generate_keyframe,
                    inputs=[
                        t4_prompt, t4_neg,
                        t4_keyframes, t4_indices, t4_img_strength,
                        t4_resolution, t4_frames, t4_fps, t4_steps, t4_seed,
                        t4_enhance, t4_fp8,
                        *t4_guidance,
                    ],
                    outputs=[t4_video, t4_info],
                )
            # ==============================================================
            # Tab 5: Audio -> Video
            # ==============================================================
            with gr.Tab("Audio → Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t5_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the video for this audio...")
                        with gr.Accordion("Negative Prompt", open=False):
                            t5_neg = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2, show_label=False)
                        t5_audio = gr.Audio(label="Audio File", type="filepath")
                        with gr.Row():
                            t5_audio_start = gr.Number(value=0.0, label="Audio Start (sec)")
                            t5_audio_max = gr.Number(value=0.0, label="Max Duration (0=all)")
                        with gr.Accordion("Conditioning Image", open=False):
                            t5_image = gr.Image(label="Image (optional)", type="numpy")
                            t5_img_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Image Strength")
                        t5_resolution = gr.Dropdown(RESOLUTION_CHOICES, value="768x512", label="Resolution (WxH)", allow_custom_value=True)
                        t5_frame_mode, t5_frames, t5_duration = create_frame_controls()
                        with gr.Row():
                            t5_fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
                            t5_steps = gr.Slider(1, 50, value=DEFAULTS.num_inference_steps, step=1, label="Steps")
                        with gr.Row():
                            t5_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        with gr.Row():
                            t5_enhance = gr.Checkbox(value=False, label="Enhance Prompt")
                            t5_fp8 = gr.Checkbox(value=True, label="FP8 Quantization")
                        t5_guidance = create_guidance_accordion("t5", show_audio=False)
                        t5_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t5_video = gr.Video(label="Generated Video")
                        t5_info = gr.Textbox(label="Info", interactive=False)
                        gr.Markdown(value=pipeline_mgr.get_loading_status, every=1)

                wire_frame_sync(t5_frame_mode, t5_frames, t5_duration, t5_fps)

                t5_btn.click(
                    fn=lambda: (None, ""),
                    outputs=[t5_video, t5_info],
                ).then(
                    fn=generate_a2vid,
                    inputs=[
                        t5_prompt, t5_neg,
                        t5_audio, t5_audio_start, t5_audio_max,
                        t5_image, t5_img_strength,
                        t5_resolution, t5_frames, t5_fps, t5_steps, t5_seed,
                        t5_enhance, t5_fp8,
                        *t5_guidance,
                    ],
                    outputs=[t5_video, t5_info],
                )
            # ==============================================================
            # Tab 6: Retake
            # ==============================================================
            with gr.Tab("Retake"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t6_video_in = gr.Video(label="Source Video", sources=["upload"])
                        t6_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the regenerated section...")
                        with gr.Accordion("Negative Prompt", open=False):
                            t6_neg = gr.Textbox(label="Negative Prompt", value="", lines=2, show_label=False)
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
                            t6_fp8 = gr.Checkbox(value=True, label="FP8 Quantization")
                        t6_guidance = create_guidance_accordion("t6")
                        t6_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t6_video_out = gr.Video(label="Result Video")
                        t6_info = gr.Textbox(label="Info", interactive=False)
                        gr.Markdown(value=pipeline_mgr.get_loading_status, every=1)

                t6_btn.click(
                    fn=lambda: (None, ""),
                    outputs=[t6_video_out, t6_info],
                ).then(
                    fn=generate_retake,
                    inputs=[
                        t6_video_in, t6_prompt, t6_neg,
                        t6_start, t6_end,
                        t6_regen_video, t6_regen_audio,
                        t6_steps, t6_seed, t6_distilled,
                        t6_enhance, t6_fp8,
                        *t6_guidance,
                    ],
                    outputs=[t6_video_out, t6_info],
                )
            # ==============================================================
            # Settings Tab
            # ==============================================================
            with gr.Tab("Settings"):
                gr.Markdown("## Model Paths")
                s_model_dir = gr.Textbox(value=pipeline_mgr.model_dir, label="Model Directory")
                with gr.Row():
                    s_apply = gr.Button("Apply", variant="secondary")
                    s_check = gr.Button("Check Models", variant="secondary")

                s_status = gr.Textbox(label="Status", interactive=False, lines=10)

                def apply_settings(model_dir):
                    pipeline_mgr.model_dir = model_dir
                    pipeline_mgr._cleanup()
                    return f"Model directory set to: {model_dir}\nPipeline cache cleared."

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
                        ("ltx-2.3-22b-ic-lora-inpainting.safetensors", "IC-LoRA Inpainting"),
                        ("ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors", "IC-LoRA Motion Track"),
                        ("ltx-2-19b-ic-lora-detailer.safetensors", "IC-LoRA Detailer"),
                        ("ltx-2-19b-ic-lora-pose-control.safetensors", "IC-LoRA Pose Control"),
                    ]

                    lines = [f"Model directory: {model_dir}\n", "=== Required ==="]
                    for fname, desc in all_files:
                        exists = (model_path / fname).exists()
                        status = "OK" if exists else "MISSING"
                        lines.append(f"  [{status}] {fname} — {desc}")

                    lines.append("\n=== IC-LoRA (optional) ===")
                    for fname, desc in ic_loras:
                        exists = (model_path / fname).exists()
                        status = "OK" if exists else "---"
                        lines.append(f"  [{status}] {fname}")

                    return "\n".join(lines)

                s_apply.click(fn=apply_settings, inputs=[s_model_dir], outputs=[s_status])
                s_check.click(fn=check_models, inputs=[s_model_dir], outputs=[s_status])

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
                    js="(()=>{if(!confirm('Delete ALL generated videos?'))throw new Error('cancelled')})",
                )

                # Auto-refresh on tab load
                app.load(fn=list_outputs, outputs=[h_file_list])

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-2 Gradio Web UI")
    parser.add_argument("--server-name", default="0.0.0.0", help="Server hostname")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    parser.add_argument("--model-dir", default=pipeline_mgr.model_dir, help="Model directory path")
    args = parser.parse_args()

    pipeline_mgr.model_dir = args.model_dir

    app = build_ui()
    app.queue()
    app.launch(server_name=args.server_name, server_port=args.server_port, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
