"""Worker process for ZIFK — GPU-heavy pipeline execution in a separate process.

Architecture:
  Main (Gradio) process                     Worker process
  ──────────────────────                     ──────────────
  WorkerProcessManager  ── task_queue ──►    _worker_loop()
                         ◄─ result_queue ──  PipelineManager
                         ◄─ progress_queue ─ (IPC progress)
"""

import logging
import multiprocessing as mp
import os
import signal
import time
import traceback
import uuid
from pathlib import Path

logger = logging.getLogger("zifk-ui")

_ctx = mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Klein inference helper (called inside worker process)
# ---------------------------------------------------------------------------
def _klein_generate(mgr, prompt, height, width, seed, num_steps=4, guidance=1.0,
                    use_cfg=False, ref_images=None):
    """Run Klein native inference. Returns PIL Image.

    Args:
        mgr: PipelineManager with klein components loaded
        prompt: text prompt
        height: output height (multiple of 16)
        width: output width (multiple of 16)
        seed: random seed
        num_steps: number of denoising steps (4 for distilled, 50 for base)
        guidance: guidance scale (1.0 for distilled, 4.0 for base)
        use_cfg: True for non-distilled models (uses denoise_cfg with empty prompt)
        ref_images: optional list of PIL Images for edit/multi-ref
    """
    import torch
    from einops import rearrange
    from PIL import Image, ExifTags

    from flux2.sampling import (
        batched_prc_img,
        batched_prc_txt,
        denoise,
        denoise_cfg,
        encode_image_refs,
        get_schedule,
        scatter_ids,
    )

    # 1. Text encode
    if use_cfg:
        # CFG: encode empty prompt + real prompt, concatenate
        ctx_empty = mgr.klein_text_encoder([""]).to(torch.bfloat16)
        ctx_prompt = mgr.klein_text_encoder([prompt]).to(torch.bfloat16)
        ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
    else:
        ctx = mgr.klein_text_encoder([prompt]).to(torch.bfloat16)
    ctx, ctx_ids = batched_prc_txt(ctx)

    # 2. Encode reference images (if any)
    ref_tokens, ref_ids = None, None
    if ref_images and len(ref_images) > 0:
        ref_tokens, ref_ids = encode_image_refs(mgr.klein_ae, ref_images)

    # 3. Generate noise
    shape = (1, 128, height // 16, width // 16)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
    x, x_ids = batched_prc_img(x)

    # 4. Schedule + denoise
    timesteps = get_schedule(num_steps, x.shape[1])

    if use_cfg:
        x = denoise_cfg(
            mgr.klein_model,
            x, x_ids,
            ctx, ctx_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )
    else:
        x = denoise(
            mgr.klein_model,
            x, x_ids,
            ctx, ctx_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )

    # 5. Decode
    x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
    x = mgr.klein_ae.decode(x).float()
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")

    # 6. Convert to PIL with EXIF
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = "AI generated;flux2;zifk"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    img.info["exif"] = exif_data.tobytes()
    return img


# ---------------------------------------------------------------------------
# Worker loop — runs in child process
# ---------------------------------------------------------------------------
def _worker_loop(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
    model_dir: str,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("zifk-worker")
    log.info("Worker started (pid=%d, model_dir=%s)", os.getpid(), model_dir)

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from pipeline_manager import PipelineManager, OUTPUT_DIR
    from zifk_config import KLEIN_BASE, KLEIN_DISTILLED

    mgr = PipelineManager(progress_queue=progress_queue)
    mgr.model_dir = model_dir

    def resolve_seed(seed):
        if seed < 0:
            return torch.randint(0, 2**31, (1,)).item()
        return int(seed)

    def make_output_path(prefix="zifk"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        return str(Path(OUTPUT_DIR) / f"{prefix}_{ts}.png")

    # -------------------------------------------------------------------
    # Handler: Z-Image T2I (Turbo or Base)
    # -------------------------------------------------------------------
    def _run_zimage_t2i(kwargs, task_id):
        from zimage import generate

        model_type = kwargs.get("model_type", "turbo")
        seed = resolve_seed(kwargs["seed"])
        mgr.load_zimage(model_type)

        # Apply attention backend if specified
        attn = kwargs.get("attention_backend")
        if attn and attn != mgr.attention_backend:
            from utils import set_attention_backend
            set_attention_backend(attn)
            mgr.attention_backend = attn

        log.info("Generating Z-Image %s %sx%s seed=%d",
                 model_type, kwargs["width"], kwargs["height"], seed)

        images = generate(
            **mgr.zimage_components,
            prompt=kwargs["prompt"],
            negative_prompt=kwargs.get("negative_prompt"),
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            num_inference_steps=int(kwargs.get("num_steps", 8 if model_type == "turbo" else 28)),
            guidance_scale=float(kwargs.get("guidance_scale", 0.0 if model_type == "turbo" else 3.5)),
            cfg_normalization=bool(kwargs.get("cfg_normalization", False)),
            cfg_truncation=float(kwargs.get("cfg_truncation", 1.0)),
            num_images_per_prompt=int(kwargs.get("num_images", 1)),
            max_sequence_length=int(kwargs.get("max_sequence_length", 512)),
            generator=torch.Generator(mgr.device).manual_seed(seed),
        )

        prefix = "zit" if model_type == "turbo" else "zib"
        paths = []
        for i, img in enumerate(images):
            p = make_output_path(f"{prefix}_{i}" if len(images) > 1 else prefix)
            img.save(p)
            paths.append(p)
        return paths, seed

    # -------------------------------------------------------------------
    # Handler: Klein T2I (Distilled)
    # -------------------------------------------------------------------
    def _run_klein_t2i(kwargs, task_id):
        seed = resolve_seed(kwargs["seed"])
        mgr.load_klein(KLEIN_DISTILLED)

        log.info("Generating Klein T2I %sx%s steps=%d guidance=%.1f seed=%d",
                 kwargs["width"], kwargs["height"],
                 kwargs.get("num_steps", 4), kwargs.get("guidance", 1.0), seed)

        img = _klein_generate(
            mgr,
            prompt=kwargs["prompt"],
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            seed=seed,
            num_steps=int(kwargs.get("num_steps", 4)),
            guidance=float(kwargs.get("guidance", 1.0)),
            use_cfg=False,
            ref_images=None,
        )

        output_path = make_output_path("klein")
        img.save(output_path, quality=95, subsampling=0)
        return output_path, seed

    # -------------------------------------------------------------------
    # Handler: Klein Base T2I (Non-distilled, CFG)
    # -------------------------------------------------------------------
    def _run_klein_base_t2i(kwargs, task_id):
        seed = resolve_seed(kwargs["seed"])
        mgr.load_klein(KLEIN_BASE)

        log.info("Generating Klein Base T2I %sx%s steps=%d guidance=%.1f seed=%d",
                 kwargs["width"], kwargs["height"],
                 kwargs.get("num_steps", 50), kwargs.get("guidance", 4.0), seed)

        img = _klein_generate(
            mgr,
            prompt=kwargs["prompt"],
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            seed=seed,
            num_steps=int(kwargs.get("num_steps", 50)),
            guidance=float(kwargs.get("guidance", 4.0)),
            use_cfg=True,
            ref_images=None,
        )

        output_path = make_output_path("klein_base")
        img.save(output_path, quality=95, subsampling=0)
        return output_path, seed

    # -------------------------------------------------------------------
    # Handler: Klein Edit (single reference)
    # -------------------------------------------------------------------
    def _run_klein_edit(kwargs, task_id):
        from PIL import Image as PILImage

        seed = resolve_seed(kwargs["seed"])
        variant = kwargs.get("klein_variant", KLEIN_DISTILLED)
        mgr.load_klein(variant)

        ref_image = PILImage.open(kwargs["image_path"]).convert("RGB")
        use_cfg = variant == KLEIN_BASE

        log.info("Generating Klein Edit %sx%s seed=%d variant=%s",
                 kwargs["width"], kwargs["height"], seed, variant)

        img = _klein_generate(
            mgr,
            prompt=kwargs["prompt"],
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            seed=seed,
            num_steps=int(kwargs.get("num_steps", 4 if not use_cfg else 50)),
            guidance=float(kwargs.get("guidance", 1.0 if not use_cfg else 4.0)),
            use_cfg=use_cfg,
            ref_images=[ref_image],
        )

        output_path = make_output_path("klein_edit")
        img.save(output_path, quality=95, subsampling=0)
        return output_path, seed

    # -------------------------------------------------------------------
    # Handler: Klein Multi-Reference
    # -------------------------------------------------------------------
    def _run_klein_multiref(kwargs, task_id):
        from PIL import Image as PILImage

        seed = resolve_seed(kwargs["seed"])
        variant = kwargs.get("klein_variant", KLEIN_DISTILLED)
        mgr.load_klein(variant)

        ref_images = []
        for p in kwargs.get("image_paths", []):
            ref_images.append(PILImage.open(p).convert("RGB"))

        use_cfg = variant == KLEIN_BASE

        log.info("Generating Klein Multi-Ref %sx%s refs=%d seed=%d",
                 kwargs["width"], kwargs["height"], len(ref_images), seed)

        img = _klein_generate(
            mgr,
            prompt=kwargs["prompt"],
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            seed=seed,
            num_steps=int(kwargs.get("num_steps", 4 if not use_cfg else 50)),
            guidance=float(kwargs.get("guidance", 1.0 if not use_cfg else 4.0)),
            use_cfg=use_cfg,
            ref_images=ref_images,
        )

        output_path = make_output_path("klein_mref")
        img.save(output_path, quality=95, subsampling=0)
        return output_path, seed

    # -------------------------------------------------------------------
    # Handler dispatch
    # -------------------------------------------------------------------
    HANDLERS = {
        "zit_t2i": _run_zimage_t2i,
        "zib_t2i": _run_zimage_t2i,
        "klein_t2i": _run_klein_t2i,
        "klein_base_t2i": _run_klein_base_t2i,
        "klein_edit": _run_klein_edit,
        "klein_multiref": _run_klein_multiref,
    }

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------
    while True:
        try:
            task = task_queue.get()
            if task is None:
                log.info("Received shutdown signal")
                break

            task_id = task["task_id"]
            gen_type = task["gen_type"]
            kwargs = task["kwargs"]
            mgr._current_task_id = task_id

            log.info("Task %s: %s started", task_id[:8], gen_type)

            handler = HANDLERS.get(gen_type)
            if handler is None:
                result_queue.put({
                    "task_id": task_id,
                    "status": "error",
                    "payload": f"Unknown generation type: {gen_type}",
                })
                continue

            try:
                result, seed = handler(kwargs, task_id)
                paths = result if isinstance(result, list) else [result]
                result_queue.put({
                    "task_id": task_id,
                    "status": "ok",
                    "payload": {"paths": paths, "seed": seed},
                })
                log.info("Task %s: completed -> %s (%d images)", task_id[:8], paths[0], len(paths))
            except Exception as e:
                log.error("Task %s failed: %s\n%s", task_id[:8], e, traceback.format_exc())
                result_queue.put({
                    "task_id": task_id,
                    "status": "error",
                    "payload": str(e),
                })
            finally:
                mgr._is_generating = False
                mgr._current_task_id = None

        except Exception as e:
            log.error("Worker loop error: %s", e)


# ---------------------------------------------------------------------------
# WorkerProcessManager — used by main (Gradio) process
# ---------------------------------------------------------------------------
class WorkerProcessManager:
    """Manages the worker subprocess."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._process: _ctx.Process | None = None
        self._task_queue: _ctx.Queue | None = None
        self._result_queue: _ctx.Queue | None = None
        self._progress_queue: _ctx.Queue | None = None

    def ensure_running(self):
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
            name="zifk-worker",
        )
        self._process.start()
        logger.info("Worker started (pid=%d)", self._process.pid)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def submit_task(self, gen_type: str, kwargs: dict) -> str:
        task_id = str(uuid.uuid4())
        self._task_queue.put({
            "task_id": task_id,
            "gen_type": gen_type,
            "kwargs": kwargs,
        })
        logger.info("Submitted task %s: %s", task_id[:8], gen_type)
        return task_id

    def poll_progress(self) -> list[dict]:
        messages = []
        while self._progress_queue is not None:
            try:
                msg = self._progress_queue.get_nowait()
                messages.append(msg)
            except Exception:
                break
        return messages

    def get_result(self, timeout: float = 0.2) -> dict | None:
        if self._result_queue is None:
            return None
        try:
            return self._result_queue.get(timeout=timeout)
        except Exception:
            return None

    def kill(self) -> str:
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

    def check_models(self, pipeline_type: str) -> list[str]:
        from pipeline_manager import REQUIRED_MODELS
        required = REQUIRED_MODELS.get(pipeline_type, [])
        missing = []
        for name in required:
            path = Path(self.model_dir) / name
            if path.is_dir():
                if not path.exists() or not any(path.rglob("*.safetensors")):
                    missing.append(name)
            else:
                if not path.exists():
                    missing.append(name)
        return missing
