"""Worker process for ZIT — GPU-heavy pipeline execution in a separate process.

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

logger = logging.getLogger("zit-ui")

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
    log = logging.getLogger("zit-worker")
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

    def make_output_path(prefix="zit"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        return str(Path(OUTPUT_DIR) / f"{prefix}_{ts}.png")

    # -------------------------------------------------------------------
    # Handler: ZIT T2I (VideoX-Fun pipeline)
    # -------------------------------------------------------------------
    def _apply_lora(kwargs):
        """Load/unload LoRA stack based on kwargs."""
        lora_stack = kwargs.get("lora_stack")
        if lora_stack:
            mgr.load_lora_stack(lora_stack)
        else:
            # Backward compatibility: single lora_name/lora_scale
            lora_name = kwargs.get("lora_name")
            lora_scale = float(kwargs.get("lora_scale", 1.0))
            if lora_name and lora_name != "None":
                mgr.load_lora_stack([{"name": lora_name, "scale": lora_scale}])
            else:
                mgr.unload_all_loras()

    def _apply_precision(kwargs, need_controlnet=True):
        """Load pipeline (BF16)."""
        mgr.load_zit(need_controlnet=need_controlnet)

    def _run_zit_t2i(kwargs, task_id):
        seed = resolve_seed(kwargs["seed"])
        _apply_precision(kwargs, need_controlnet=False)
        _apply_lora(kwargs)

        pipeline = mgr.zit_components["pipeline"]

        # Set time_shift on scheduler
        time_shift = float(kwargs.get("time_shift", 3.0))
        pipeline.scheduler = type(pipeline.scheduler).from_config(
            pipeline.scheduler.config, shift=time_shift,
        )

        log.info("Generating ZIT %sx%s seed=%d steps=%d shift=%.1f",
                 kwargs["width"], kwargs["height"], seed,
                 int(kwargs.get("num_steps", 8)), time_shift)

        result = pipeline(
            prompt=kwargs["prompt"],
            negative_prompt=kwargs.get("negative_prompt"),
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            num_inference_steps=int(kwargs.get("num_steps", 8)),
            guidance_scale=float(kwargs.get("guidance_scale", 1.0)),
            cfg_normalization=bool(kwargs.get("cfg_normalization", False)),
            cfg_truncation=float(kwargs.get("cfg_truncation", 1.0)),
            num_images_per_prompt=int(kwargs.get("num_images", 1)),
            max_sequence_length=int(kwargs.get("max_sequence_length", 512)),
            generator=torch.Generator(mgr.device).manual_seed(seed),
            # No control_image → pure T2I
        )

        images = result.images
        paths = []
        for i, img in enumerate(images):
            p = make_output_path(f"zit_{i}" if len(images) > 1 else "zit")
            img.save(p)
            paths.append(p)
        return paths, seed

    # -------------------------------------------------------------------
    # Handler: ControlNet (Canny/Pose/Depth/HED/Scribble/Gray)
    # -------------------------------------------------------------------
    def _run_controlnet(kwargs, task_id):
        from PIL import Image as PILImage

        seed = resolve_seed(kwargs["seed"])
        _apply_precision(kwargs)
        _apply_lora(kwargs)

        pipeline = mgr.zit_components["pipeline"]

        time_shift = float(kwargs.get("time_shift", 3.0))
        pipeline.scheduler = type(pipeline.scheduler).from_config(
            pipeline.scheduler.config, shift=time_shift,
        )

        # Load pre-processed control image
        control_img = PILImage.open(kwargs["control_image_path"]).convert("RGB")

        # Load original image (if provided) for style/color harmony
        original_img = None
        if kwargs.get("original_image_path"):
            original_img = PILImage.open(kwargs["original_image_path"]).convert("RGB")

        log.info("ControlNet %sx%s mode=%s scale=%.2f seed=%d original=%s",
                 kwargs["width"], kwargs["height"],
                 kwargs.get("control_mode", "?"),
                 float(kwargs.get("control_scale", 0.7)), seed,
                 "yes" if original_img else "no")

        result = pipeline(
            prompt=kwargs["prompt"],
            negative_prompt=kwargs.get("negative_prompt"),
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            image=original_img,
            control_image=control_img,
            control_context_scale=float(kwargs.get("control_scale", 0.7)),
            control_step_cutoff=float(kwargs.get("control_step_cutoff", 0.5)),
            num_inference_steps=int(kwargs.get("num_steps", 8)),
            guidance_scale=float(kwargs.get("guidance_scale", 1.0)),
            cfg_normalization=bool(kwargs.get("cfg_normalization", False)),
            cfg_truncation=float(kwargs.get("cfg_truncation", 1.0)),
            num_images_per_prompt=int(kwargs.get("num_images", 1)),
            max_sequence_length=int(kwargs.get("max_sequence_length", 512)),
            generator=torch.Generator(mgr.device).manual_seed(seed),
        )

        images = result.images
        mode_tag = kwargs.get("control_mode", "ctrl")
        paths = []
        for i, img in enumerate(images):
            p = make_output_path(f"cn_{mode_tag}_{i}" if len(images) > 1 else f"cn_{mode_tag}")
            img.save(p, quality=95)
            paths.append(p)
        return paths, seed

    # -------------------------------------------------------------------
    # Handler: Inpaint
    # -------------------------------------------------------------------
    def _run_inpaint(kwargs, task_id):
        from PIL import Image as PILImage

        seed = resolve_seed(kwargs["seed"])
        need_cn = kwargs.get("need_controlnet", True)
        _apply_precision(kwargs, need_controlnet=need_cn)
        _apply_lora(kwargs)

        pipeline = mgr.zit_components["pipeline"]

        time_shift = float(kwargs.get("time_shift", 3.0))
        pipeline.scheduler = type(pipeline.scheduler).from_config(
            pipeline.scheduler.config, shift=time_shift,
        )

        # Load image and mask
        image = PILImage.open(kwargs["image_path"]).convert("RGB")
        mask = PILImage.open(kwargs["mask_path"]).convert("L")

        log.info("Inpaint %sx%s scale=%.2f guidance=%.1f steps=%d seed=%d",
                 kwargs["width"], kwargs["height"],
                 float(kwargs.get("control_scale", 0.7)),
                 float(kwargs.get("guidance_scale", 1.0)),
                 int(kwargs.get("num_steps", 8)), seed)

        # control_image: None for inpaint (aistudynow workflow)
        # Optional: control_image_path for explicit pose/canny/depth
        if kwargs.get("control_image_path"):
            ctrl_img = PILImage.open(kwargs["control_image_path"]).convert("RGB")
            log.info("Inpaint: using explicit control_image from %s", kwargs["control_image_path"])
        else:
            ctrl_img = None
        # Load separate DiffDiff mask if provided
        dd_mask = None
        if kwargs.get("diffdiff_mask_path"):
            dd_mask = PILImage.open(kwargs["diffdiff_mask_path"]).convert("L")

        result = pipeline(
            prompt=kwargs["prompt"],
            negative_prompt=kwargs.get("negative_prompt"),
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            image=image,
            mask_image=mask,
            diffdiff_mask=dd_mask,
            control_image=ctrl_img,
            control_context_scale=float(kwargs.get("control_scale", 0.7)),
            control_step_cutoff=float(kwargs.get("control_step_cutoff", 0.5)),
            denoise=float(kwargs.get("denoise", 1.0)),
            num_inference_steps=int(kwargs.get("num_steps", 8)),
            guidance_scale=float(kwargs.get("guidance_scale", 1.0)),
            cfg_truncation=float(kwargs.get("cfg_truncation", 1.0)),
            max_sequence_length=int(kwargs.get("max_sequence_length", 512)),
            generator=torch.Generator(mgr.device).manual_seed(seed),
        )

        output_path = make_output_path("inpaint")
        result.images[0].save(output_path, quality=95)
        return output_path, seed

    # -------------------------------------------------------------------
    # Handler: Outpaint (Crop & Stitch inpaint approach)
    # Expand canvas → crop boundary region → inpaint as "surrounded" mask
    # → stitch back. Treats outpaint as inpaint so mask has context on all sides.
    # -------------------------------------------------------------------
    def _run_outpaint(kwargs, task_id):
        """JoPD-style outpainting: PadImage + GrowMaskWithBlur + single inpaint."""
        import numpy as np
        import cv2
        import tempfile
        from PIL import Image as PILImage

        seed = resolve_seed(kwargs["seed"])
        direction = kwargs.get("direction", "Right")
        dirs = direction if isinstance(direction, list) else [direction]
        expand_px = int(kwargs.get("expand_px", 256))

        log.info("Outpaint direction=%s expand=%dpx seed=%d", dirs, expand_px, seed)

        image = PILImage.open(kwargs["image_path"]).convert("RGB")
        w, h = image.size
        img_arr = np.array(image)

        # --- Step 1: Pad Image — replicate edge pixels for tone continuity ---
        pad = {"Left": 0, "Right": 0, "Up": 0, "Down": 0}
        for d in dirs:
            pad[d] = expand_px

        new_w = w + pad["Left"] + pad["Right"]
        new_h = h + pad["Up"] + pad["Down"]

        canvas = cv2.copyMakeBorder(
            img_arr, pad["Up"], pad["Down"], pad["Left"], pad["Right"],
            cv2.BORDER_REPLICATE,
        )

        # Binary mask: 255=expansion (generate), 0=original (keep)
        binary_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        if pad["Right"] > 0:
            binary_mask[:, pad["Left"] + w:] = 255
        if pad["Left"] > 0:
            binary_mask[:, :pad["Left"]] = 255
        if pad["Down"] > 0:
            binary_mask[pad["Up"] + h:, :] = 255
        if pad["Up"] > 0:
            binary_mask[:pad["Up"], :] = 255

        # --- Step 2: GrowMaskWithBlur (JoPD core technique) ---
        mask_grow = int(kwargs.get("mask_grow", 40))
        mask_blur = int(kwargs.get("mask_blur", 30))

        if mask_grow > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * mask_grow + 1, 2 * mask_grow + 1)
            )
            mask = cv2.dilate(binary_mask, kernel, iterations=1)
        else:
            mask = binary_mask.copy()

        if mask_blur > 0:
            blur_k = mask_blur * 2 + 1
            mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

        log.info("Outpaint: %sx%s -> %sx%s GrowMask(grow=%d, blur=%d)",
                 w, h, new_w, new_h, mask_grow, mask_blur)

        # --- Step 3: Single inpaint pass ---
        gen_w = (new_w // 16) * 16
        gen_h = (new_h // 16) * 16

        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
        PILImage.fromarray(canvas).save(tmp_img.name)
        tmp_img.close()
        tmp_mask = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
        PILImage.fromarray(mask).save(tmp_mask.name)
        tmp_mask.close()

        inpaint_kwargs = {
            "prompt": kwargs["prompt"],
            "negative_prompt": kwargs.get("negative_prompt"),
            "image_path": tmp_img.name,
            "mask_path": tmp_mask.name,
            "width": gen_w, "height": gen_h,
            "seed": seed,
            "denoise": float(kwargs.get("denoise", 1.0)),
            "num_steps": int(kwargs.get("num_steps", 8)),
            "guidance_scale": float(kwargs.get("guidance_scale", 1.0)),
            "cfg_truncation": float(kwargs.get("cfg_truncation", 1.0)),
            "control_scale": float(kwargs.get("control_scale", 0.7)),
            "control_step_cutoff": float(kwargs.get("control_step_cutoff", 0.7)),
            "time_shift": float(kwargs.get("time_shift", 3.0)),
            "max_sequence_length": int(kwargs.get("max_sequence_length", 512)),
            "need_controlnet": True,
        }
        if kwargs.get("lora_stack"):
            inpaint_kwargs["lora_stack"] = kwargs["lora_stack"]

        result_path, _ = _run_inpaint(inpaint_kwargs, task_id)
        result_path = result_path if isinstance(result_path, str) else result_path[0]

        for f in [tmp_img.name, tmp_mask.name]:
            try: Path(f).unlink(missing_ok=True)
            except: pass

        result_img = PILImage.open(result_path).convert("RGB")
        result_arr = np.array(result_img)
        if result_arr.shape[0] != new_h or result_arr.shape[1] != new_w:
            result_arr = cv2.resize(result_arr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Remove intermediate inpaint file (outpaint saves its own)
        try:
            Path(result_path).unlink(missing_ok=True)
            json_p = Path(result_path).with_suffix(".json")
            if json_p.exists():
                json_p.unlink()
        except Exception:
            pass

        output_path = make_output_path("outpaint")
        PILImage.fromarray(result_arr).save(output_path, quality=95)
        log.info("Outpaint: complete -> %s", output_path)

        return output_path, seed

    # -------------------------------------------------------------------
    # Handler dispatch
    # -------------------------------------------------------------------
    HANDLERS = {
        "zit_t2i": _run_zit_t2i,
        "controlnet": _run_controlnet,
        "inpaint": _run_inpaint,
        "outpaint": _run_outpaint,
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
                tb = traceback.format_exc()
                log.error("Task %s failed: %s\n%s", task_id[:8], e, tb)
                result_queue.put({
                    "task_id": task_id,
                    "status": "error",
                    "payload": f"{e}\n\n{tb}",
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
        self._current_need_controlnet: bool | None = None  # track ControlNet mode

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
            name="zit-worker",
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
        self._current_need_controlnet = None  # reset mode tracking
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

    def ensure_mode(self, need_controlnet: bool):
        """Kill and restart worker if ControlNet mode changed.

        Avoids peak memory from old+new transformer coexisting during
        in-process reload (~97GB on unified memory).
        """
        if self._current_need_controlnet is not None and \
           self._current_need_controlnet != need_controlnet and \
           self.is_alive():
            logger.info("ControlNet mode changed (%s → %s), killing worker for clean restart",
                        "with CN" if self._current_need_controlnet else "without CN",
                        "with CN" if need_controlnet else "without CN")
            self.kill()
        self._current_need_controlnet = need_controlnet

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
