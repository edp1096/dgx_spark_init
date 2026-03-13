"""Video generation functions — thin shims that submit tasks to the worker process."""

import logging
import tempfile
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from pipeline_manager import IC_LORA_MAP, OUTPUT_DIR, REQUIRED_MODELS
from worker import WorkerProcessManager

logger = logging.getLogger("ltx2-ui")

# ---------------------------------------------------------------------------
# Worker process singleton
# ---------------------------------------------------------------------------
_worker_mgr: WorkerProcessManager | None = None


def get_worker_mgr() -> WorkerProcessManager:
    global _worker_mgr
    if _worker_mgr is None:
        from config import MODEL_DIR
        _worker_mgr = WorkerProcessManager(str(MODEL_DIR))
    return _worker_mgr


def set_model_dir(model_dir: str):
    """Update model dir — kills worker so next generation uses new dir."""
    mgr = get_worker_mgr()
    mgr.model_dir = model_dir
    if mgr.is_alive():
        mgr.kill()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_temp_image(image_array) -> str:
    """Save numpy image array to a temp file. Returns path."""
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=OUTPUT_DIR)
    Image.fromarray(image_array).save(f.name)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Shared loading status (updated from progress_queue)
# ---------------------------------------------------------------------------
_loading_status = ""
_loading_plan: list[str] = []       # full load order from worker
_loading_done: dict[str, float] = {}  # name -> elapsed seconds
_loading_current: str | None = None   # currently loading model name
_last_enhanced_prompt: str = ""


def get_loading_status() -> str:
    """Called by gr.Markdown(every=1) to show progress."""
    return _loading_status


# ---------------------------------------------------------------------------
# Generation result tracking (for UI reconnection after browser refresh)
# ---------------------------------------------------------------------------
_gen_active = False
_last_gen_result: dict | None = None   # {"path", "info", "gen_type", "time", "consumed"}
_result_version = 0


def get_result_version() -> int:
    """Polled by hidden UI monitor. Increments on each generation completion."""
    return _result_version


def get_last_gen_result() -> dict | None:
    return _last_gen_result


def consume_last_gen_result() -> None:
    """Mark the last result as consumed so it won't be restored on refresh."""
    if _last_gen_result is not None:
        _last_gen_result["consumed"] = True


def is_generation_active() -> bool:
    return _gen_active


def get_gen_info_for_tab(gen_type: str) -> str:
    """Return info text for a specific tab. Safe for every=N polling."""
    if _gen_active:
        snap = _active_gen_inputs
        if snap and snap["gen_type"] == gen_type:
            return "Generating..."
        return ""
    result = _last_gen_result
    if result is None:
        return ""
    if result["gen_type"] != gen_type:
        return ""
    if time.time() - result["time"] > 600:
        return ""
    return result["info"]


# ---------------------------------------------------------------------------
# Active generation input snapshot (for refresh-persistence)
# ---------------------------------------------------------------------------
_active_gen_inputs: dict | None = None  # {"gen_type": str, "values": list}


def get_active_gen_inputs() -> dict | None:
    """Return saved inputs if generation is active, else None."""
    if _gen_active and _active_gen_inputs:
        return _active_gen_inputs
    return None


def _build_loading_status(plan: list[str], done: dict[str, float], current: str | None) -> str:
    """Build Markdown status showing all model loading steps."""
    if not plan:
        return ""
    total = len(plan)
    done_count = len(done)
    lines = []
    for name in plan:
        if name in done:
            elapsed = done[name]
            if elapsed == 0:
                lines.append(f"- **{name}**: cached")
            else:
                lines.append(f"- **{name}**: ok ({elapsed:.0f}s)")
        elif name == current:
            lines.append(f"- **{name}**: loading...")
        else:
            lines.append(f"- {name}: wait")
    header = f"**Model Loading [{done_count}/{total}]**"
    if done_count == total:
        total_time = sum(done.values())
        if total_time > 0:
            header += f" — done in {total_time:.0f}s"
        else:
            header += " — all cached"
    return header + "\n" + "\n".join(lines)


def _submit_and_wait(gen_type: str, kwargs: dict, progress) -> tuple[str, str]:
    """Submit task to worker, poll progress, wait for result."""
    global _loading_status, _loading_plan, _loading_done, _loading_current
    global _gen_active, _last_gen_result, _result_version, _last_enhanced_prompt

    _gen_active = True
    _last_enhanced_prompt = ""

    mgr = get_worker_mgr()
    mgr.ensure_running()
    task_id = mgr.submit_task(gen_type, kwargs)

    _loading_status = "**Starting generation...**"
    _loading_plan = []
    _loading_done = {}
    _loading_current = None
    start = time.time()

    try:
        while True:
            # Check worker alive
            if not mgr.is_alive():
                _loading_status = ""
                raise gr.Error("Worker process crashed. Click Generate to restart.")

            # Drain progress messages
            for msg in mgr.poll_progress():
                if msg.get("task_id") != task_id:
                    continue
                mtype = msg.get("type")
                data = msg.get("data", {})

                if mtype == "loading_plan":
                    _loading_plan = data.get("plan", [])
                    _loading_done = {}
                    _loading_current = None
                    _loading_status = _build_loading_status(_loading_plan, _loading_done, _loading_current)

                elif mtype == "loading_start":
                    name = data.get("name", "?")
                    idx = data.get("index", 0)
                    total = data.get("total", 1)
                    _loading_current = name
                    if progress:
                        progress((idx - 1) / max(total, 1), desc=f"Loading {name} ({idx}/{total})")
                    _loading_status = _build_loading_status(_loading_plan, _loading_done, _loading_current)

                elif mtype == "loading_done":
                    name = data.get("name", "?")
                    idx = data.get("index", 0)
                    total = data.get("total", 1)
                    elapsed = data.get("elapsed", 0)
                    _loading_done[name] = elapsed
                    _loading_current = None
                    if progress:
                        progress(idx / max(total, 1), desc=f"Loading {name} ({idx}/{total})")
                    _loading_status = _build_loading_status(_loading_plan, _loading_done, _loading_current)

                elif mtype == "loading":
                    # Legacy fallback
                    name = data.get("name", "?")
                    idx = data.get("index", 0)
                    total = data.get("total", 1)
                    elapsed = data.get("elapsed", 0)
                    _loading_done[name] = elapsed
                    _loading_current = None
                    if progress:
                        progress(idx / max(total, 1), desc=f"Loading {name} ({idx}/{total})")
                    _loading_status = _build_loading_status(_loading_plan, _loading_done, _loading_current)

                elif mtype == "enhanced_prompt":
                    _last_enhanced_prompt = data.get("text", "")

                elif mtype == "stage1_preview":
                    preview_path = data.get("path")
                    if preview_path:
                        _loading_status = "**Upscaling (Stage 2)...**"
                        yield preview_path, "⏳ Stage 1 미리보기 (절반 해상도) — 업스케일 진행 중...", _last_enhanced_prompt

                elif mtype == "step":
                    current = data.get("current", 0)
                    total = data.get("total", 1)
                    desc = data.get("desc", "Denoising")
                    if progress and total > 0:
                        progress(current / max(total, 1), desc=f"{desc} [{current}/{total}]")
                    _loading_status = f"**Denoising** [{current}/{total}] {desc}"

            # Check for result (short timeout)
            result = mgr.get_result(timeout=0.2)
            if result and result.get("task_id") == task_id:
                elapsed = time.time() - start
                _loading_status = ""

                if result["status"] == "ok":
                    payload = result["payload"]
                    path = payload["path"]
                    seed = payload["seed"]
                    info = f"Seed: {seed} | Time: {elapsed:.1f}s | Output: {Path(path).name}"
                    logger.info("Generation complete: %s", info)
                    _gen_active = False
                    _active_gen_inputs = None
                    _result_version += 1
                    _last_gen_result = {
                        "path": path, "info": info,
                        "gen_type": gen_type, "time": time.time(),
                        "consumed": False,
                    }
                    yield path, info, _last_enhanced_prompt
                    return
                elif result["status"] == "cancelled":
                    raise gr.Error("Generation cancelled.")
                else:
                    raise gr.Error(f"Generation failed: {result['payload']}")
    except Exception:
        _gen_active = False
        _active_gen_inputs = None
        _loading_status = ""
        raise


def _validate(pipeline_type: str, prompt: str, required_files: dict | None = None,
              resolution: str | None = None):
    """Input validation (runs in main process, no GPU needed)."""
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is required.")

    mgr = get_worker_mgr()
    missing = mgr.check_models(pipeline_type)
    if missing:
        raise gr.Error(f"Missing model files: {', '.join(missing)}\nCheck Settings tab for model directory.")

    if required_files:
        for name, value in required_files.items():
            if value is None:
                raise gr.Error(f"{name} is required.")



# ---------------------------------------------------------------------------
# Generation functions — each one validates, packs kwargs, calls _submit_and_wait
# ---------------------------------------------------------------------------
def generate_ti2vid(
    prompt, negative_prompt, image, image_strength, image_crf,
    resolution, num_frames, frame_rate, num_steps, seed, sampler,
    enhance_prompt, fp8,
    v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks, v_skip_step,
    a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks, a_skip_step,
    frame_mode="Frames", duration=4.8, disable_audio=False,
    progress=gr.Progress(track_tqdm=True),
):
    global _active_gen_inputs
    _active_gen_inputs = {"gen_type": "ti2vid", "values": [
        prompt, negative_prompt, image, image_strength, image_crf,
        resolution, num_frames, frame_rate, num_steps, seed, sampler,
        enhance_prompt, fp8,
        v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks, v_skip_step,
        a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks, a_skip_step,
        frame_mode, duration, disable_audio,
    ]}
    _validate("ti2vid", prompt, resolution=resolution)
    image_path = save_temp_image(image) if image is not None else None
    kwargs = {
        "prompt": prompt, "negative_prompt": negative_prompt,
        "image_path": image_path, "image_strength": float(image_strength),
        "image_crf": int(image_crf), "resolution": resolution,
        "num_frames": int(num_frames), "frame_rate": int(frame_rate),
        "num_steps": int(num_steps), "seed": int(seed), "sampler": sampler,
        "enhance_prompt": bool(enhance_prompt), "fp8": bool(fp8),
        "v_guidance": [v_cfg, v_stg, v_rescale, v_modality, str(v_stg_blocks), v_skip_step],
        "a_guidance": [a_cfg, a_stg, a_rescale, a_modality, str(a_stg_blocks), a_skip_step],
        "disable_audio": bool(disable_audio),
    }
    yield from _submit_and_wait("ti2vid", kwargs, progress)


def generate_distilled(
    prompt, negative_prompt, nag_scale,
    image, image_strength, image_crf,
    resolution, num_frames, frame_rate, seed,
    enhance_prompt, fp8,
    frame_mode="Frames", duration=4.8, disable_audio=False,
    progress=gr.Progress(track_tqdm=True),
):
    global _active_gen_inputs
    _active_gen_inputs = {"gen_type": "distilled", "values": [
        prompt, negative_prompt, nag_scale,
        image, image_strength, image_crf,
        resolution, num_frames, frame_rate, seed,
        enhance_prompt, fp8,
        frame_mode, duration, disable_audio,
    ]}
    _validate("distilled", prompt)
    image_path = save_temp_image(image) if image is not None else None
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt, "nag_scale": float(nag_scale),
        "image_path": image_path, "image_strength": float(image_strength),
        "image_crf": int(image_crf), "resolution": resolution,
        "num_frames": int(num_frames), "frame_rate": int(frame_rate),
        "seed": int(seed),
        "enhance_prompt": bool(enhance_prompt), "fp8": bool(fp8),
        "disable_audio": bool(disable_audio),
    }
    yield from _submit_and_wait("distilled", kwargs, progress)


def generate_iclora(
    prompt, negative_prompt, nag_scale,
    ref_video, ref_strength, lora_choice, attention_strength,
    image, image_strength, image_crf,
    resolution, num_frames, frame_rate, seed,
    skip_stage2, enhance_prompt, fp8,
    frame_mode="Frames", duration=4.8, disable_audio=False,
    progress=gr.Progress(track_tqdm=True),
):
    global _active_gen_inputs
    _active_gen_inputs = {"gen_type": "iclora", "values": [
        prompt, negative_prompt, nag_scale,
        ref_video, ref_strength, lora_choice, attention_strength,
        image, image_strength, image_crf,
        resolution, num_frames, frame_rate, seed,
        skip_stage2, enhance_prompt, fp8,
        frame_mode, duration, disable_audio,
    ]}
    _validate("iclora", prompt, required_files={"Reference Video": ref_video})

    # Check LoRA file exists
    lora_filename = IC_LORA_MAP.get(lora_choice, IC_LORA_MAP["Union Control"])
    mgr = get_worker_mgr()
    lora_path = Path(mgr.model_dir) / lora_filename
    if not lora_path.exists():
        raise gr.Error(f"IC-LoRA file not found: {lora_filename}")

    image_path = save_temp_image(image) if image is not None else None
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt, "nag_scale": float(nag_scale),
        "ref_video": ref_video,
        "ref_strength": float(ref_strength),
        "lora_choice": lora_choice, "attention_strength": float(attention_strength),
        "image_path": image_path, "image_strength": float(image_strength),
        "image_crf": int(image_crf), "resolution": resolution,
        "num_frames": int(num_frames), "frame_rate": int(frame_rate),
        "seed": int(seed), "skip_stage2": bool(skip_stage2),
        "enhance_prompt": bool(enhance_prompt), "fp8": bool(fp8),
        "disable_audio": bool(disable_audio),
    }
    yield from _submit_and_wait("iclora", kwargs, progress)


def generate_keyframe(
    prompt, negative_prompt,
    keyframe_files, frame_indices_str, image_strength, image_crf,
    resolution, num_frames, frame_rate, num_steps, seed,
    enhance_prompt, fp8,
    v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks, v_skip_step,
    a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks, a_skip_step,
    frame_mode="Frames", duration=4.8, disable_audio=False,
    progress=gr.Progress(track_tqdm=True),
):
    global _active_gen_inputs
    _active_gen_inputs = {"gen_type": "keyframe", "values": [
        prompt, negative_prompt,
        keyframe_files, frame_indices_str, image_strength, image_crf,
        resolution, num_frames, frame_rate, num_steps, seed,
        enhance_prompt, fp8,
        v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks, v_skip_step,
        a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks, a_skip_step,
        frame_mode, duration, disable_audio,
    ]}
    if not keyframe_files or len(keyframe_files) < 2:
        raise gr.Error("At least 2 keyframe images are required.")
    _validate("keyframe", prompt, resolution=resolution)
    kwargs = {
        "prompt": prompt, "negative_prompt": negative_prompt,
        "keyframe_paths": [str(f) for f in keyframe_files],
        "frame_indices": frame_indices_str,
        "image_strength": float(image_strength), "image_crf": int(image_crf),
        "resolution": resolution,
        "num_frames": int(num_frames), "frame_rate": int(frame_rate),
        "num_steps": int(num_steps), "seed": int(seed),
        "enhance_prompt": bool(enhance_prompt), "fp8": bool(fp8),
        "v_guidance": [v_cfg, v_stg, v_rescale, v_modality, str(v_stg_blocks), v_skip_step],
        "a_guidance": [a_cfg, a_stg, a_rescale, a_modality, str(a_stg_blocks), a_skip_step],
        "disable_audio": bool(disable_audio),
    }
    yield from _submit_and_wait("keyframe", kwargs, progress)


def generate_a2vid(
    prompt, negative_prompt,
    audio_file, audio_start, audio_max_duration,
    image, image_strength, image_crf,
    resolution, num_frames, frame_rate, num_steps, seed,
    enhance_prompt, fp8,
    v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks, v_skip_step,
    frame_mode="Frames", duration=4.8,
    progress=gr.Progress(track_tqdm=True),
):
    global _active_gen_inputs
    _active_gen_inputs = {"gen_type": "a2vid", "values": [
        prompt, negative_prompt,
        audio_file, audio_start, audio_max_duration,
        image, image_strength, image_crf,
        resolution, num_frames, frame_rate, num_steps, seed,
        enhance_prompt, fp8,
        v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks, v_skip_step,
        frame_mode, duration,
    ]}
    _validate("a2vid", prompt, required_files={"Audio File": audio_file}, resolution=resolution)
    image_path = save_temp_image(image) if image is not None else None
    kwargs = {
        "prompt": prompt, "negative_prompt": negative_prompt,
        "audio_path": audio_file, "audio_start": float(audio_start),
        "audio_max_duration": float(audio_max_duration),
        "image_path": image_path, "image_strength": float(image_strength),
        "image_crf": int(image_crf), "resolution": resolution,
        "num_frames": int(num_frames), "frame_rate": int(frame_rate),
        "num_steps": int(num_steps), "seed": int(seed),
        "enhance_prompt": bool(enhance_prompt), "fp8": bool(fp8),
        "v_guidance": [v_cfg, v_stg, v_rescale, v_modality, str(v_stg_blocks), v_skip_step],
    }
    yield from _submit_and_wait("a2vid", kwargs, progress)


def generate_retake(
    video_path, prompt, negative_prompt, nag_scale,
    start_time, end_time,
    regenerate_video, regenerate_audio,
    num_steps, seed, distilled_mode,
    enhance_prompt, fp8,
    v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks, v_skip_step,
    a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks, a_skip_step,
    progress=gr.Progress(track_tqdm=True),
):
    global _active_gen_inputs
    _active_gen_inputs = {"gen_type": "retake", "values": [
        video_path, prompt, negative_prompt, nag_scale,
        start_time, end_time,
        regenerate_video, regenerate_audio,
        num_steps, seed, distilled_mode,
        enhance_prompt, fp8,
        v_cfg, v_stg, v_rescale, v_modality, v_stg_blocks, v_skip_step,
        a_cfg, a_stg, a_rescale, a_modality, a_stg_blocks, a_skip_step,
    ]}
    if start_time >= end_time:
        raise gr.Error("Start time must be less than end time.")
    _validate("retake", prompt, required_files={"Source Video": video_path})
    kwargs = {
        "prompt": prompt, "negative_prompt": negative_prompt,
        "nag_scale": float(nag_scale),
        "video_path": video_path,
        "start_time": float(start_time), "end_time": float(end_time),
        "regenerate_video": bool(regenerate_video),
        "regenerate_audio": bool(regenerate_audio),
        "num_steps": int(num_steps), "seed": int(seed),
        "distilled": bool(distilled_mode),
        "enhance_prompt": bool(enhance_prompt), "fp8": bool(fp8),
        "v_guidance": [v_cfg, v_stg, v_rescale, v_modality, str(v_stg_blocks), v_skip_step],
        "a_guidance": [a_cfg, a_stg, a_rescale, a_modality, str(a_stg_blocks), a_skip_step],
    }
    yield from _submit_and_wait("retake", kwargs, progress)
