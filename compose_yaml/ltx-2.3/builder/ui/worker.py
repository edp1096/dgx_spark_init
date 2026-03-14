"""Worker process for video generation with emergency kill support.

Runs the GPU-heavy pipeline in a separate process so that the main Gradio
process can kill it instantly (process.kill()) when the user hits the
emergency stop button.  Communication uses multiprocessing Queues.

Architecture:
  Main (Gradio) process                     Worker process
  ──────────────────────                     ──────────────
  WorkerProcessManager  ── task_queue ──►    _worker_loop()
                         ◄─ result_queue ──  PipelineManager
                         ◄─ progress_queue ─ (IPC progress)
"""

import logging
import multiprocessing as mp
import warnings

warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
import os
import signal
import time
import traceback
import uuid
from pathlib import Path

logger = logging.getLogger("ltx2-ui")

# Use spawn context to avoid CUDA fork issues
_ctx = mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Worker loop — runs in child process
# ---------------------------------------------------------------------------
def _worker_loop(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
    model_dir: str,
):
    """Main loop for the worker process. Blocks on task_queue, executes generation."""
    # Re-seed logging in child
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ltx2-worker")
    log.info("Worker started (pid=%d, model_dir=%s)", os.getpid(), model_dir)

    # Ignore SIGINT in worker — let main process handle Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Lazy imports — heavy GPU libs only in worker
    import torch
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    # --- Performance optimizations ---
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from pipeline_manager import PipelineManager, IC_LORA_MAP, OUTPUT_DIR
    from mod.nag import encode_negative_prompt, get_model_ledger, nag_guidance

    # --- Qwen prompt enhancement + vision (replaces Gemma monkey-patch) ---
    _qwen_model = None
    _qwen_processor = None

    # System prompts from LTX's Gemma prompts
    _prompts_dir = (Path(__file__).resolve().parent.parent
        / "LTX-2" / "packages" / "ltx-core" / "src" / "ltx_core"
        / "text_encoders" / "gemma" / "encoders" / "prompts")
    _QWEN_T2V_SYSTEM_PROMPT = (_prompts_dir / "gemma_t2v_system_prompt.txt").read_text()
    _QWEN_I2V_SYSTEM_PROMPT = (_prompts_dir / "gemma_i2v_system_prompt.txt").read_text()

    def _load_qwen():
        nonlocal _qwen_model, _qwen_processor
        if _qwen_model is not None:
            return
        from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        qwen_path = str(Path(model_dir) / "Huihui-Qwen3.5-4B-abliterated")
        log.info("Loading Qwen3.5-4B VLM 8bit (%s)...", qwen_path)
        _qwen_processor = AutoProcessor.from_pretrained(qwen_path)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        _qwen_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            qwen_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        log.info("Qwen3.5-4B VLM loaded (8bit, ~4.5GB)")

    def _qwen_generate(messages, images=None, max_new_tokens=512):
        """Shared generate helper for Qwen VLM (text-only or vision)."""
        _load_qwen()
        text = _qwen_processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = _qwen_processor(
            text=text,
            images=images,
            return_tensors="pt",
        ).to(_qwen_model.device)
        with torch.inference_mode():
            outputs = _qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
            )
        generated = outputs[0][len(inputs.input_ids[0]):]
        result = _qwen_processor.tokenizer.decode(generated, skip_special_tokens=True)
        return result.strip()

    def _enhance_prompt_qwen(prompt: str) -> str:
        """Enhance a text prompt using Qwen3.5-4B VLM."""
        messages = [
            {"role": "system", "content": _QWEN_T2V_SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Output ONLY the enhanced prompt as a single paragraph. "
                "Do NOT include any thinking process, reasoning, analysis, or step-by-step explanation.\n"
                "IMPORTANT: Preserve all character dialogue/speech EXACTLY in its original language. "
                "Do NOT translate dialogue into English. For example, if dialogue is in Korean, keep it in Korean within quotes.\n\n"
                f"user prompt: {prompt}"
            )},
        ]
        enhanced = _qwen_generate(messages)
        # Strip thinking/reasoning text: take from "Style:" if present
        style_idx = enhanced.find("Style:")
        if style_idx > 0:
            enhanced = enhanced[style_idx:]
            log.info("Stripped %d chars of thinking text before 'Style:'", style_idx)
        return enhanced

    def _describe_image_qwen(image_path: str, hint: str = "") -> str:
        """Describe an image using Qwen3.5-4B VLM for prompt suggestion."""
        from PIL import Image as PILImage
        image = PILImage.open(image_path).convert("RGB")
        user_text = hint if hint.strip() else "Describe this image in detail for video generation."
        messages = [
            {"role": "system", "content": _QWEN_I2V_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"User Raw Input Prompt: {user_text}"},
            ]},
        ]
        return _qwen_generate(messages, images=[image])

    log.info("Qwen3.5-4B VLM prompt enhancement configured (lazy load)")

    mgr = PipelineManager(progress_queue=progress_queue)

    def make_preview_callback(task_id, fps, num_frames):
        """Create a callback that saves Stage 1 preview and sends it via progress_queue."""
        def callback(video_frames, audio, frame_rate, n_frames):
            try:
                preview_path = str(Path(OUTPUT_DIR) / f"_preview_{task_id[:8]}.mp4")
                encode_video(
                    video=video_frames, fps=fps,
                    audio=audio, output_path=preview_path,
                    video_chunks_number=get_video_chunks_number(num_frames),
                )
                progress_queue.put_nowait({
                    "task_id": task_id,
                    "type": "stage1_preview",
                    "data": {"path": preview_path},
                })
                log.info("Task %s: Stage 1 preview saved -> %s", task_id[:8], preview_path)
            except Exception as e:
                log.warning("Task %s: Stage 1 preview failed: %s", task_id[:8], e)
        return callback
    mgr.model_dir = model_dir

    def parse_resolution(res_str):
        w, h = res_str.split("x")
        w, h = round(int(w) / 64) * 64, round(int(h) / 64) * 64
        return h, w

    def resolve_seed(seed):
        if seed < 0:
            return torch.randint(0, 2**31, (1,)).item()
        return int(seed)

    def make_output_path():
        ts = time.strftime("%Y%m%d_%H%M%S")
        return str(Path(OUTPUT_DIR) / f"ltx2_{ts}.mp4")

    def build_guider(params):
        cfg, stg, rescale, modality, stg_blocks_str, skip_step = params
        stg_blocks = [int(x.strip()) for x in str(stg_blocks_str).split(",") if x.strip()]
        return MultiModalGuiderParams(
            cfg_scale=float(cfg),
            stg_scale=float(stg),
            rescale_scale=float(rescale),
            modality_scale=float(modality),
            stg_blocks=stg_blocks,
            skip_step=int(skip_step),
        )

    # --- NAG helper ---
    def _run_pipeline_with_nag(pipeline, kwargs, gen_kwargs):
        """Run pipeline with optional NAG guidance for distilled models."""
        neg_prompt = kwargs.get("negative_prompt", "")
        nag_scale = kwargs.get("nag_scale", 1.0)

        if neg_prompt.strip() and nag_scale > 1.0:
            nag_alpha = kwargs.get("nag_alpha", 0.0)
            ledger = get_model_ledger(pipeline)
            neg_v, neg_a = encode_negative_prompt(ledger, neg_prompt)
            with nag_guidance(neg_v, neg_a, scale=nag_scale, alpha=nag_alpha):
                return pipeline(**gen_kwargs)
        return pipeline(**gen_kwargs)

    # --- Image conditioning helper ---
    def _build_images(kwargs):
        """Build ImageConditioningInput list from kwargs."""
        conds = kwargs.get("image_conditionings", [])
        if not conds:
            # legacy fallback
            if kwargs.get("image_path"):
                return [ImageConditioningInput(
                    kwargs["image_path"], 0,
                    kwargs["image_strength"], kwargs["image_crf"])]
            return []
        return [ImageConditioningInput(c["path"], c["frame_idx"], c["strength"], c["crf"])
                for c in conds]

    # --- Generation handlers ---
    def _run_ti2vid(kwargs, task_id):
        with torch.inference_mode():
            seed = resolve_seed(kwargs["seed"])
            height, width = parse_resolution(kwargs["resolution"])
            sampler = kwargs.get("sampler", "euler")
            lora_strength = kwargs.get("lora_strength", 0.8)
            custom_loras = kwargs.get("custom_loras", [])
            pipeline = mgr.get_ti2vid(sampler=sampler, lora_strength=lora_strength,
                                      custom_loras=custom_loras, quantization="fp8")

            images = _build_images(kwargs)

            video_guider = build_guider(kwargs["v_guidance"])
            audio_guider = build_guider(kwargs["a_guidance"])

            mgr.start_loading_bar()
            gen_kwargs = dict(
                prompt=kwargs["prompt"],
                negative_prompt=kwargs.get("negative_prompt", ""),
                seed=seed,
                height=height, width=width,
                num_frames=kwargs["num_frames"],
                frame_rate=kwargs["frame_rate"],
                num_inference_steps=kwargs["num_steps"],
                video_guider_params=video_guider,
                audio_guider_params=audio_guider,
                images=images,
                enhance_prompt=kwargs.get("enhance_prompt", False),
            )
            gen_kwargs["stage1_preview_callback"] = make_preview_callback(
                task_id, kwargs["frame_rate"], kwargs["num_frames"])

            video_frames, audio = pipeline(**gen_kwargs)
            if kwargs.get("disable_audio"):
                audio = None
            output_path = make_output_path()
            encode_video(
                video=video_frames, fps=kwargs["frame_rate"],
                audio=audio, output_path=output_path,
                video_chunks_number=get_video_chunks_number(kwargs["num_frames"]),
            )
            return output_path, seed

    def _run_distilled(kwargs, task_id):
        with torch.inference_mode():
            seed = resolve_seed(kwargs["seed"])
            height, width = parse_resolution(kwargs["resolution"])
            pipeline = mgr.get_distilled(quantization="fp8")

            images = _build_images(kwargs)

            mgr.start_loading_bar()
            gen_kwargs = dict(
                prompt=kwargs["prompt"],
                seed=seed,
                height=height, width=width,
                num_frames=kwargs["num_frames"],
                frame_rate=kwargs["frame_rate"],
                images=images,
                enhance_prompt=kwargs.get("enhance_prompt", False),
            )
            gen_kwargs["stage1_preview_callback"] = make_preview_callback(
                task_id, kwargs["frame_rate"], kwargs["num_frames"])

            video_frames, audio = _run_pipeline_with_nag(pipeline, kwargs, gen_kwargs)
            if kwargs.get("disable_audio"):
                audio = None
            output_path = make_output_path()
            encode_video(
                video=video_frames, fps=kwargs["frame_rate"],
                audio=audio, output_path=output_path,
                video_chunks_number=get_video_chunks_number(kwargs["num_frames"]),
            )
            return output_path, seed

    def _run_iclora(kwargs, task_id):
        with torch.inference_mode():
            seed = resolve_seed(kwargs["seed"])
            height, width = parse_resolution(kwargs["resolution"])

            lora_choices = kwargs.get("lora_choices", [])
            if not lora_choices:
                # Backward compat: single lora_choice string
                choice = kwargs.get("lora_choice", "Union Control")
                lora_choices = [choice]
            lora_paths = [str(Path(mgr.model_dir) / IC_LORA_MAP.get(c, IC_LORA_MAP["Union Control"]))
                          for c in lora_choices]
            distilled_lora_strength = kwargs.get("lora_strength", 0.8)
            custom_loras = kwargs.get("custom_loras", [])
            pipeline = mgr.get_iclora(
                lora_paths=lora_paths,
                distilled_lora_strength=distilled_lora_strength,
                custom_loras=custom_loras,
                quantization="fp8",
            )

            images = _build_images(kwargs)

            video_conditioning = []
            if kwargs.get("ref_video"):
                video_conditioning = [(kwargs["ref_video"], kwargs["ref_strength"])]

            mgr.start_loading_bar()
            gen_kwargs = dict(
                prompt=kwargs["prompt"],
                seed=seed,
                height=height, width=width,
                num_frames=kwargs["num_frames"],
                frame_rate=kwargs["frame_rate"],
                images=images,
                video_conditioning=video_conditioning,
                conditioning_attention_strength=kwargs.get("attention_strength", 1.0),
                skip_stage_2=kwargs.get("skip_stage2", False),
                enhance_prompt=kwargs.get("enhance_prompt", False),
            )
            if not kwargs.get("skip_stage2", False):
                gen_kwargs["stage1_preview_callback"] = make_preview_callback(
                    task_id, kwargs["frame_rate"], kwargs["num_frames"])

            video_frames, audio = _run_pipeline_with_nag(pipeline, kwargs, gen_kwargs)
            if kwargs.get("disable_audio"):
                audio = None
            output_path = make_output_path()
            encode_video(
                video=video_frames, fps=kwargs["frame_rate"],
                audio=audio, output_path=output_path,
                video_chunks_number=get_video_chunks_number(kwargs["num_frames"]),
            )
            return output_path, seed

    def _run_keyframe(kwargs, task_id):
        with torch.inference_mode():
            seed = resolve_seed(kwargs["seed"])
            height, width = parse_resolution(kwargs["resolution"])
            lora_strength = kwargs.get("lora_strength", 0.8)
            custom_loras = kwargs.get("custom_loras", [])
            pipeline = mgr.get_keyframe(lora_strength=lora_strength,
                                        custom_loras=custom_loras, quantization="fp8")

            images = []
            crf = kwargs["image_crf"]
            for kc in kwargs["keyframe_conditionings"]:
                images.append(ImageConditioningInput(
                    kc["path"], kc["frame_idx"], kc["strength"], crf,
                ))

            video_guider = build_guider(kwargs["v_guidance"])
            audio_guider = build_guider(kwargs["a_guidance"])

            mgr.start_loading_bar()
            gen_kwargs = dict(
                prompt=kwargs["prompt"],
                negative_prompt=kwargs.get("negative_prompt", ""),
                seed=seed,
                height=height, width=width,
                num_frames=kwargs["num_frames"],
                frame_rate=kwargs["frame_rate"],
                num_inference_steps=kwargs["num_steps"],
                video_guider_params=video_guider,
                audio_guider_params=audio_guider,
                images=images,
                enhance_prompt=kwargs.get("enhance_prompt", False),
            )
            gen_kwargs["stage1_preview_callback"] = make_preview_callback(
                task_id, kwargs["frame_rate"], kwargs["num_frames"])

            video_frames, audio = pipeline(**gen_kwargs)
            if kwargs.get("disable_audio"):
                audio = None
            output_path = make_output_path()
            encode_video(
                video=video_frames, fps=kwargs["frame_rate"],
                audio=audio, output_path=output_path,
                video_chunks_number=get_video_chunks_number(kwargs["num_frames"]),
            )
            return output_path, seed

    def _preprocess_audio(audio_path, num_frames, frame_rate):
        """Preprocess audio for the LTX audio encoder.

        Handles three issues that cause latent shape mismatches:
        1. Channel count: encoder expects stereo (2ch)
        2. Sample rate: encoder expects 16 kHz
        3. Duration: audio must be >= video duration, pad with silence if shorter
        """
        import soundfile as sf
        import numpy as np

        TARGET_SR = 16000
        video_duration = num_frames / frame_rate + 1.0  # +1s safety margin

        # Try native soundfile first, fall back to ffmpeg for mp3 etc.
        try:
            data, sr = sf.read(audio_path)
        except Exception:
            import subprocess, tempfile
            tmp_wav = str(Path(OUTPUT_DIR) / "_ffmpeg_tmp.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", str(TARGET_SR),
                 "-ac", "2", "-f", "wav", tmp_wav],
                capture_output=True, check=True,
            )
            data, sr = sf.read(tmp_wav)

        # Ensure 2D array (samples, channels)
        if data.ndim == 1:
            data = np.stack([data, data], axis=-1)
        elif data.shape[-1] != 2:
            # Downmix N channels to stereo (take first two or average)
            if data.shape[-1] >= 2:
                data = data[:, :2]
            else:
                data = np.stack([data[:, 0], data[:, 0]], axis=-1)

        # Resample to 16 kHz if needed
        if sr != TARGET_SR:
            from scipy.signal import resample
            num_samples_new = int(len(data) * TARGET_SR / sr)
            data = resample(data, num_samples_new, axis=0).astype(np.float64)
            sr = TARGET_SR

        # Pad with silence if audio is shorter than video
        required_samples = int(video_duration * sr)
        if len(data) < required_samples:
            pad = np.zeros((required_samples - len(data), 2), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=0)
            log.info("Padded audio with %.1fs silence (video=%.1fs)",
                     (required_samples - len(data)) / sr, video_duration)

        out_path = str(Path(OUTPUT_DIR) / "_preproc_audio.wav")
        sf.write(out_path, data, sr, subtype="PCM_16")
        log.info("Preprocessed audio: %s → %s (%d ch, %d Hz, %.1fs)",
                 audio_path, out_path, data.shape[-1], sr, len(data) / sr)
        return out_path

    def _run_a2vid(kwargs, task_id):
        with torch.inference_mode():
            seed = resolve_seed(kwargs["seed"])
            height, width = parse_resolution(kwargs["resolution"])
            lora_strength = kwargs.get("lora_strength", 0.8)
            custom_loras = kwargs.get("custom_loras", [])
            pipeline = mgr.get_a2vid(lora_strength=lora_strength,
                                     custom_loras=custom_loras, quantization="fp8")

            audio_path = _preprocess_audio(
                kwargs["audio_path"], kwargs["num_frames"], kwargs["frame_rate"]
            )

            images = _build_images(kwargs)

            video_guider = build_guider(kwargs["v_guidance"])
            audio_max = kwargs["audio_max_duration"] if kwargs["audio_max_duration"] > 0 else None

            mgr.start_loading_bar()
            video_frames, audio = pipeline(
                prompt=kwargs["prompt"],
                negative_prompt=kwargs.get("negative_prompt", ""),
                seed=seed,
                height=height, width=width,
                num_frames=kwargs["num_frames"],
                frame_rate=kwargs["frame_rate"],
                num_inference_steps=kwargs["num_steps"],
                video_guider_params=video_guider,
                images=images,
                audio_path=audio_path,
                audio_start_time=kwargs.get("audio_start", 0),
                audio_max_duration=audio_max,
                enhance_prompt=kwargs.get("enhance_prompt", False),
                stage1_preview_callback=make_preview_callback(
                    task_id, kwargs["frame_rate"], kwargs["num_frames"]),
            )
            output_path = make_output_path()
            encode_video(
                video=video_frames, fps=kwargs["frame_rate"],
                audio=audio, output_path=output_path,
                video_chunks_number=get_video_chunks_number(kwargs["num_frames"]),
            )
            return output_path, seed

    def _run_retake(kwargs, task_id):
        with torch.inference_mode():
            seed = resolve_seed(kwargs["seed"])
            distilled = kwargs.get("distilled", False)
            pipeline = mgr.get_retake(distilled=distilled, quantization="fp8")

            video_guider = build_guider(kwargs["v_guidance"])
            audio_guider = build_guider(kwargs["a_guidance"])

            mgr.start_loading_bar()
            retake_kwargs = dict(
                video_path=kwargs["video_path"],
                prompt=kwargs["prompt"],
                start_time=kwargs["start_time"],
                end_time=kwargs["end_time"],
                seed=seed,
                negative_prompt=kwargs.get("negative_prompt", ""),
                num_inference_steps=kwargs["num_steps"],
                video_guider_params=video_guider if not distilled else None,
                audio_guider_params=audio_guider if not distilled else None,
                regenerate_video=kwargs.get("regenerate_video", True),
                regenerate_audio=kwargs.get("regenerate_audio", True),
                enhance_prompt=kwargs.get("enhance_prompt", False),
                distilled=distilled,
            )
            if distilled:
                video_frames, audio_tensor = _run_pipeline_with_nag(
                    pipeline, kwargs, retake_kwargs,
                )
            else:
                video_frames, audio_tensor = pipeline(**retake_kwargs)
            from ltx_pipelines.utils.media_io import get_videostream_metadata
            src_fps, src_num_frames, _, _ = get_videostream_metadata(kwargs["video_path"])
            output_path = make_output_path()
            encode_video(
                video=video_frames, fps=src_fps,
                audio=audio_tensor, output_path=output_path,
                video_chunks_number=get_video_chunks_number(src_num_frames),
            )
            return output_path, seed

    def _run_describe_frame(kwargs, task_id):
        """Use Qwen3.5 VLM to describe an image for prompt suggestion."""
        image_path = kwargs["image_path"]
        hint = kwargs.get("hint", "")
        description = _describe_image_qwen(image_path, hint)
        log.info("Task %s: Qwen described frame: %s", task_id[:8], description[:100])
        return description, 0

    HANDLERS = {
        "ti2vid": _run_ti2vid,
        "distilled": _run_distilled,
        "iclora": _run_iclora,
        "keyframe": _run_keyframe,
        "a2vid": _run_a2vid,
        "retake": _run_retake,
        "describe_frame": _run_describe_frame,
    }

    # --- Main loop ---
    while True:
        try:
            task = task_queue.get()  # blocks
            if task is None:
                log.info("Received shutdown signal")
                break

            task_id = task["task_id"]
            gen_type = task["gen_type"]
            kwargs = task["kwargs"]

            mgr._current_task_id = task_id
            log.info("Task %s: %s started", task_id[:8], gen_type)

            # --- Qwen prompt enhancement (before pipeline call, skip for describe_frame) ---
            if gen_type != "describe_frame" and kwargs.get("enhance_prompt", False):
                try:
                    original = kwargs["prompt"]
                    enhanced = _enhance_prompt_qwen(original)
                    kwargs["prompt"] = enhanced
                    kwargs["enhance_prompt"] = False  # bypass Gemma enhancement
                    log.info("Task %s: Qwen enhanced prompt: %s", task_id[:8], enhanced[:100])
                    progress_queue.put_nowait({
                        "task_id": task_id,
                        "type": "enhanced_prompt",
                        "data": {"text": enhanced},
                    })
                except Exception as e:
                    log.warning("Task %s: Qwen enhancement failed (%s), using original prompt", task_id[:8], e)
                    kwargs["enhance_prompt"] = False

            handler = HANDLERS.get(gen_type)
            if handler is None:
                result_queue.put({
                    "task_id": task_id,
                    "status": "error",
                    "payload": f"Unknown generation type: {gen_type}",
                })
                continue

            try:
                path, seed = handler(kwargs, task_id)
                result_queue.put({
                    "task_id": task_id,
                    "status": "ok",
                    "payload": {"path": path, "seed": seed},
                })
                log.info("Task %s: completed -> %s", task_id[:8], path)
            except Exception as e:
                log.error("Task %s failed: %s\n%s", task_id[:8], e, traceback.format_exc())
                result_queue.put({
                    "task_id": task_id,
                    "status": "error",
                    "payload": str(e),
                })
            finally:
                mgr.stop_loading_bar()
                mgr._current_task_id = None

        except Exception as e:
            log.error("Worker loop error: %s", e)


# ---------------------------------------------------------------------------
# WorkerProcessManager — used by main (Gradio) process
# ---------------------------------------------------------------------------
class WorkerProcessManager:
    """Manages the worker subprocess. Provides submit/poll/kill interface."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._process: _ctx.Process | None = None
        self._task_queue: _ctx.Queue | None = None
        self._result_queue: _ctx.Queue | None = None
        self._progress_queue: _ctx.Queue | None = None

    def ensure_running(self):
        """Start worker process if not already running."""
        if self._process is not None and self._process.is_alive():
            return
        logger.info("Starting worker process (model_dir=%s)", self.model_dir)
        self._task_queue = _ctx.Queue()
        self._result_queue = _ctx.Queue()
        self._progress_queue = _ctx.Queue()
        self._process = _ctx.Process(
            target=_worker_loop,
            args=(self._task_queue, self._result_queue, self._progress_queue, self.model_dir),
            daemon=True,
            name="ltx2-worker",
        )
        self._process.start()
        logger.info("Worker started (pid=%d)", self._process.pid)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def submit_task(self, gen_type: str, kwargs: dict) -> str:
        """Submit a generation task. Returns task_id."""
        task_id = str(uuid.uuid4())
        self._task_queue.put({
            "task_id": task_id,
            "gen_type": gen_type,
            "kwargs": kwargs,
        })
        logger.info("Submitted task %s: %s", task_id[:8], gen_type)
        return task_id

    def poll_progress(self) -> list[dict]:
        """Drain all pending progress messages (non-blocking)."""
        messages = []
        while self._progress_queue is not None:
            try:
                msg = self._progress_queue.get_nowait()
                messages.append(msg)
            except Exception:
                break
        return messages

    def get_result(self, timeout: float = 0.2) -> dict | None:
        """Check for a result (with short timeout)."""
        if self._result_queue is None:
            return None
        try:
            return self._result_queue.get(timeout=timeout)
        except Exception:
            return None

    def kill(self) -> str:
        """Emergency kill — SIGKILL the worker process."""
        if self._process is None or not self._process.is_alive():
            return "No active worker process."
        pid = self._process.pid
        logger.warning("KILLING worker process (pid=%d)", pid)
        self._process.kill()
        self._process.join(timeout=5)
        self._process = None
        self._task_queue = None
        self._result_queue = None
        self._progress_queue = None
        return f"Worker killed (pid={pid}). Will restart on next generation."

    def stop(self):
        """Graceful shutdown — send None sentinel and wait."""
        if self._process is None or not self._process.is_alive():
            return
        logger.info("Stopping worker gracefully...")
        try:
            self._task_queue.put(None)
            self._process.join(timeout=10)
        except Exception:
            pass
        if self._process is not None and self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=5)
        self._process = None
        self._task_queue = None
        self._result_queue = None
        self._progress_queue = None
        logger.info("Worker stopped.")

    def check_models(self, pipeline_type: str) -> list[str]:
        """Check for missing model files (runs in main process, no GPU needed)."""
        from pipeline_manager import REQUIRED_MODELS
        required = REQUIRED_MODELS.get(pipeline_type, [])
        missing = []
        for f in required:
            if not (Path(self.model_dir) / f).exists():
                missing.append(f)
        return missing
