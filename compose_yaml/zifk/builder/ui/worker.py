"""Worker process for Z-Image generation with emergency kill support.

Runs the GPU-heavy pipeline in a separate process so that the main Gradio
process can kill it instantly when the user hits the emergency stop button.
Communication uses multiprocessing Queues.

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

logger = logging.getLogger("zimage-ui")

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("zimage-worker")
    log.info("Worker started (pid=%d, model_dir=%s)", os.getpid(), model_dir)

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from pipeline_manager import PipelineManager, OUTPUT_DIR

    mgr = PipelineManager(progress_queue=progress_queue)
    mgr.model_dir = model_dir

    def resolve_seed(seed):
        if seed < 0:
            return torch.randint(0, 2**31, (1,)).item()
        return int(seed)

    def make_output_path():
        ts = time.strftime("%Y%m%d_%H%M%S")
        return str(Path(OUTPUT_DIR) / f"zimage_{ts}.png")

    def _run_turbo(kwargs, task_id):
        seed = resolve_seed(kwargs["seed"])
        loras = kwargs.get("loras", [])
        pipe = mgr.get_turbo(loras=loras)

        log.info("Generating Turbo %sx%s seed=%d", kwargs["width"], kwargs["height"], seed)
        result = pipe(
            prompt=kwargs["prompt"],
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            num_inference_steps=int(kwargs.get("num_steps", 9)),
            guidance_scale=0.0,
            num_images_per_prompt=int(kwargs.get("num_images", 1)),
            generator=torch.Generator(mgr.device).manual_seed(seed),
        )
        output_path = make_output_path()
        result.images[0].save(output_path)
        return output_path, seed

    def _run_base(kwargs, task_id):
        seed = resolve_seed(kwargs["seed"])
        loras = kwargs.get("loras", [])
        pipe = mgr.get_base(loras=loras)

        log.info("Generating Base %sx%s steps=%d cfg=%.1f seed=%d",
                 kwargs["width"], kwargs["height"], kwargs["num_steps"],
                 kwargs["guidance_scale"], seed)
        result = pipe(
            prompt=kwargs["prompt"],
            negative_prompt=kwargs.get("negative_prompt", ""),
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            num_inference_steps=int(kwargs["num_steps"]),
            guidance_scale=float(kwargs["guidance_scale"]),
            cfg_normalization=bool(kwargs.get("cfg_normalization", False)),
            cfg_truncation=float(kwargs.get("cfg_truncation", 1.0)),
            num_images_per_prompt=int(kwargs.get("num_images", 1)),
            generator=torch.Generator(mgr.device).manual_seed(seed),
        )
        output_path = make_output_path()
        result.images[0].save(output_path)
        return output_path, seed

    def _run_img2img(kwargs, task_id):
        from PIL import Image as PILImage
        seed = resolve_seed(kwargs["seed"])
        use_base = kwargs.get("use_base", False)
        loras = kwargs.get("loras", [])
        pipe = mgr.get_img2img(use_base=use_base, loras=loras)

        init_image = PILImage.open(kwargs["image_path"]).convert("RGB")
        log.info("Generating Img2Img %sx%s strength=%.2f seed=%d",
                 kwargs["width"], kwargs["height"], kwargs["strength"], seed)

        gen_kwargs = dict(
            prompt=kwargs["prompt"],
            image=init_image,
            strength=float(kwargs["strength"]),
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            num_inference_steps=int(kwargs["num_steps"]),
            generator=torch.Generator(mgr.device).manual_seed(seed),
        )
        if use_base:
            gen_kwargs["negative_prompt"] = kwargs.get("negative_prompt", "")
            gen_kwargs["guidance_scale"] = float(kwargs.get("guidance_scale", 4.0))

        result = pipe(**gen_kwargs)
        output_path = make_output_path()
        result.images[0].save(output_path)
        return output_path, seed

    HANDLERS = {
        "turbo": _run_turbo,
        "base": _run_base,
        "img2img": _run_img2img,
    }

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
            name="zimage-worker",
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
        for d in required:
            path = Path(self.model_dir) / d
            if not path.exists() or not any(path.rglob("*.safetensors")):
                missing.append(d)
        return missing
