"""Video generation functions — bridge between UI inputs and pipeline calls."""

import logging
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae import get_video_chunks_number
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

from pipeline_manager import IC_LORA_MAP, OUTPUT_DIR, pipeline_mgr

logger = logging.getLogger("ltx2-ui")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse 'WxH' string → (height, width), rounded to nearest multiple of 64."""
    w, h = res_str.split("x")
    w, h = round(int(w) / 64) * 64, round(int(h) / 64) * 64
    return h, w


def make_output_path() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return str(Path(OUTPUT_DIR) / f"ltx2_{ts}.mp4")


def resolve_seed(seed: int) -> int:
    if seed < 0:
        return torch.randint(0, 2**31, (1,)).item()
    return int(seed)


def build_guider(cfg_scale, stg_scale, rescale_scale, modality_scale, stg_blocks_str) -> MultiModalGuiderParams:
    stg_blocks = [int(x.strip()) for x in stg_blocks_str.split(",") if x.strip()]
    return MultiModalGuiderParams(
        cfg_scale=cfg_scale,
        stg_scale=stg_scale,
        rescale_scale=rescale_scale,
        modality_scale=modality_scale,
        stg_blocks=stg_blocks,
    )


def save_temp_image(image_array) -> str:
    """Save numpy image array to a temp file. Returns path."""
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=OUTPUT_DIR)
    Image.fromarray(image_array).save(f.name)
    f.close()
    return f.name


def validate_and_run(pipeline_type: str, prompt: str, generate_fn, required_files=None, progress=None):
    """Wrapper: validate inputs, check models, run generation, handle errors."""
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is required.")

    missing = pipeline_mgr.check_models(pipeline_type)
    if missing:
        raise gr.Error(f"Missing model files: {', '.join(missing)}\nCheck Settings tab for model directory.")

    if required_files:
        for name, value in required_files.items():
            if value is None:
                raise gr.Error(f"{name} is required.")

    start = time.time()
    try:
        result_path, seed = generate_fn(progress)
        elapsed = time.time() - start
        info = f"Seed: {seed} | Time: {elapsed:.1f}s | Output: {Path(result_path).name}"
        logger.info("Generation complete: %s", info)
        return result_path, info
    except gr.Error:
        raise
    except Exception as e:
        logger.error("Generation failed: %s\n%s", e, traceback.format_exc())
        raise gr.Error(f"Generation failed: {e}")
    finally:
        pipeline_mgr.stop_loading_bar()


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------
def generate_ti2vid(
    prompt, negative_prompt, image, image_strength,
    resolution, num_frames, frame_rate, num_steps, seed, sampler,
    enhance_prompt, fp8,
    v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks,
    a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks,
    progress=gr.Progress(track_tqdm=True),
):
    def _generate(prog):
        with torch.inference_mode():
            _seed = resolve_seed(seed)
            height, width = parse_resolution(resolution)
            quantization = "fp8" if fp8 else None
            pipeline = pipeline_mgr.get_ti2vid(sampler=sampler, quantization=quantization)

            images = []
            if image is not None:
                images = [ImageConditioningInput(save_temp_image(image), 0, image_strength, 33)]

            video_guider = build_guider(v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks)
            audio_guider = build_guider(a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks)

            pipeline_mgr.start_loading_bar()
            video_frames, audio = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=_seed,
                height=height, width=width,
                num_frames=int(num_frames), frame_rate=frame_rate,
                num_inference_steps=num_steps,
                video_guider_params=video_guider,
                audio_guider_params=audio_guider,
                images=images,
                enhance_prompt=enhance_prompt,
            )
            if prog:
                prog(0.9, desc="Encoding video...")
            output_path = make_output_path()
            encode_video(video=video_frames, fps=frame_rate, audio=audio, output_path=output_path,
                         video_chunks_number=get_video_chunks_number(num_frames))
            return output_path, _seed

    return validate_and_run("ti2vid", prompt, _generate, progress=progress)


def generate_distilled(
    prompt, image, image_strength,
    resolution, num_frames, frame_rate, seed,
    enhance_prompt, fp8,
    progress=gr.Progress(track_tqdm=True),
):
    def _generate(prog):
        with torch.inference_mode():
            _seed = resolve_seed(seed)
            height, width = parse_resolution(resolution)
            quantization = "fp8" if fp8 else None
            pipeline = pipeline_mgr.get_distilled(quantization=quantization)

            images = []
            if image is not None:
                images = [ImageConditioningInput(save_temp_image(image), 0, image_strength, 33)]

            pipeline_mgr.start_loading_bar()
            video_frames, audio = pipeline(
                prompt=prompt,
                seed=_seed,
                height=height, width=width,
                num_frames=int(num_frames), frame_rate=frame_rate,
                images=images,
                enhance_prompt=enhance_prompt,
            )
            if prog:
                prog(0.9, desc="Encoding video...")
            output_path = make_output_path()
            encode_video(video=video_frames, fps=frame_rate, audio=audio, output_path=output_path,
                         video_chunks_number=get_video_chunks_number(num_frames))
            return output_path, _seed

    return validate_and_run("distilled", prompt, _generate, progress=progress)


def generate_iclora(
    prompt, ref_video, ref_strength, lora_choice, attention_strength,
    image, image_strength,
    resolution, num_frames, frame_rate, seed,
    skip_stage2, enhance_prompt, fp8,
    progress=gr.Progress(track_tqdm=True),
):
    def _generate(prog):
        with torch.inference_mode():
            _seed = resolve_seed(seed)
            height, width = parse_resolution(resolution)

            lora_filename = IC_LORA_MAP.get(lora_choice, IC_LORA_MAP["Union Control"])
            lora_path = str(Path(pipeline_mgr.model_dir) / lora_filename)
            if not Path(lora_path).exists():
                raise gr.Error(f"IC-LoRA file not found: {lora_filename}")

            quantization = "fp8" if fp8 else None
            pipeline = pipeline_mgr.get_iclora(lora_path=lora_path, quantization=quantization)

            images = []
            if image is not None:
                images = [ImageConditioningInput(save_temp_image(image), 0, image_strength, 33)]

            video_conditioning = []
            if ref_video is not None:
                video_conditioning = [(ref_video, ref_strength)]

            pipeline_mgr.start_loading_bar()
            video_frames, audio = pipeline(
                prompt=prompt,
                seed=_seed,
                height=height, width=width,
                num_frames=int(num_frames), frame_rate=frame_rate,
                images=images,
                video_conditioning=video_conditioning,
                conditioning_attention_strength=attention_strength,
                skip_stage_2=skip_stage2,
                enhance_prompt=enhance_prompt,
            )
            if prog:
                prog(0.9, desc="Encoding video...")
            output_path = make_output_path()
            encode_video(video=video_frames, fps=frame_rate, audio=audio, output_path=output_path,
                         video_chunks_number=get_video_chunks_number(num_frames))
            return output_path, _seed

    return validate_and_run("iclora", prompt, _generate,
                            required_files={"Reference Video": ref_video}, progress=progress)


def generate_keyframe(
    prompt, negative_prompt,
    keyframe_files, frame_indices_str, image_strength,
    resolution, num_frames, frame_rate, num_steps, seed,
    enhance_prompt, fp8,
    v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks,
    a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks,
    progress=gr.Progress(track_tqdm=True),
):
    if not keyframe_files or len(keyframe_files) < 2:
        raise gr.Error("At least 2 keyframe images are required.")

    def _generate(prog):
        with torch.inference_mode():
            _seed = resolve_seed(seed)
            height, width = parse_resolution(resolution)
            quantization = "fp8" if fp8 else None
            pipeline = pipeline_mgr.get_keyframe(quantization=quantization)

            indices = [int(x.strip()) for x in frame_indices_str.split(",") if x.strip()]
            images = []
            for i, kf in enumerate(keyframe_files):
                idx = indices[i] if i < len(indices) else i * (num_frames // max(len(keyframe_files) - 1, 1))
                images.append(ImageConditioningInput(kf, idx, image_strength, 33))

            video_guider = build_guider(v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks)
            audio_guider = build_guider(a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks)

            pipeline_mgr.start_loading_bar()
            video_frames, audio = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=_seed,
                height=height, width=width,
                num_frames=int(num_frames), frame_rate=frame_rate,
                num_inference_steps=num_steps,
                video_guider_params=video_guider,
                audio_guider_params=audio_guider,
                images=images,
                enhance_prompt=enhance_prompt,
            )
            if prog:
                prog(0.9, desc="Encoding video...")
            output_path = make_output_path()
            encode_video(video=video_frames, fps=frame_rate, audio=audio, output_path=output_path,
                         video_chunks_number=get_video_chunks_number(num_frames))
            return output_path, _seed

    return validate_and_run("keyframe", prompt, _generate, progress=progress)


def generate_a2vid(
    prompt, negative_prompt,
    audio_file, audio_start, audio_max_duration,
    image, image_strength,
    resolution, num_frames, frame_rate, num_steps, seed,
    enhance_prompt, fp8,
    v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks,
    progress=gr.Progress(track_tqdm=True),
):
    def _generate(prog):
        with torch.inference_mode():
            _seed = resolve_seed(seed)
            height, width = parse_resolution(resolution)
            quantization = "fp8" if fp8 else None
            pipeline = pipeline_mgr.get_a2vid(quantization=quantization)

            images = []
            if image is not None:
                images = [(save_temp_image(image), 0, image_strength)]

            video_guider = build_guider(v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks)
            audio_max = audio_max_duration if audio_max_duration > 0 else None

            pipeline_mgr.start_loading_bar()
            video_frames, audio = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=_seed,
                height=height, width=width,
                num_frames=int(num_frames), frame_rate=frame_rate,
                num_inference_steps=num_steps,
                video_guider_params=video_guider,
                images=images,
                audio_path=audio_file,
                audio_start_time=audio_start,
                audio_max_duration=audio_max,
                enhance_prompt=enhance_prompt,
            )
            if prog:
                prog(0.9, desc="Encoding video...")
            output_path = make_output_path()
            encode_video(video=video_frames, fps=frame_rate, audio=audio, output_path=output_path,
                         video_chunks_number=get_video_chunks_number(num_frames))
            return output_path, _seed

    return validate_and_run("a2vid", prompt, _generate,
                            required_files={"Audio File": audio_file}, progress=progress)


def generate_retake(
    video_path, prompt, negative_prompt,
    start_time, end_time,
    regenerate_video, regenerate_audio,
    num_steps, seed, distilled_mode,
    enhance_prompt, fp8,
    v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks,
    a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks,
    progress=gr.Progress(track_tqdm=True),
):
    if start_time >= end_time:
        raise gr.Error("Start time must be less than end time.")

    def _generate(prog):
        with torch.inference_mode():
            _seed = resolve_seed(seed)
            quantization = "fp8" if fp8 else None
            pipeline = pipeline_mgr.get_retake(distilled=distilled_mode, quantization=quantization)

            video_guider = build_guider(v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks)
            audio_guider = build_guider(a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks)

            pipeline_mgr.start_loading_bar()
            video_frames, audio_tensor = pipeline(
                video_path=video_path,
                prompt=prompt,
                start_time=start_time,
                end_time=end_time,
                seed=_seed,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                video_guider_params=video_guider if not distilled_mode else None,
                audio_guider_params=audio_guider if not distilled_mode else None,
                regenerate_video=regenerate_video,
                regenerate_audio=regenerate_audio,
                enhance_prompt=enhance_prompt,
                distilled=distilled_mode,
            )
            if prog:
                prog(0.9, desc="Encoding video...")
            from ltx_pipelines.utils.media_io import get_videostream_metadata
            src_fps, src_num_frames, _, _ = get_videostream_metadata(video_path)
            output_path = make_output_path()
            encode_video(video=video_frames, fps=src_fps, audio=audio_tensor, output_path=output_path,
                         video_chunks_number=get_video_chunks_number(src_num_frames))
            return output_path, _seed

    return validate_and_run("retake", prompt, _generate,
                            required_files={"Source Video": video_path}, progress=progress)
