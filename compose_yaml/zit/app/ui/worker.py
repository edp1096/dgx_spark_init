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
    def _run_zit_t2i(kwargs, task_id):
        seed = resolve_seed(kwargs["seed"])
        mgr.load_zit()

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
            guidance_scale=float(kwargs.get("guidance_scale", 0.5)),
            cfg_normalization=bool(kwargs.get("cfg_normalization", False)),
            cfg_truncation=float(kwargs.get("cfg_truncation", 0.9)),
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
        mgr.load_zit()

        pipeline = mgr.zit_components["pipeline"]

        time_shift = float(kwargs.get("time_shift", 3.0))
        pipeline.scheduler = type(pipeline.scheduler).from_config(
            pipeline.scheduler.config, shift=time_shift,
        )

        # Load pre-processed control image
        control_img = PILImage.open(kwargs["control_image_path"]).convert("RGB")

        log.info("ControlNet %sx%s mode=%s scale=%.2f seed=%d",
                 kwargs["width"], kwargs["height"],
                 kwargs.get("control_mode", "?"),
                 float(kwargs.get("control_scale", 0.65)), seed)

        result = pipeline(
            prompt=kwargs["prompt"],
            negative_prompt=kwargs.get("negative_prompt"),
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            control_image=control_img,
            control_context_scale=float(kwargs.get("control_scale", 0.65)),
            num_inference_steps=int(kwargs.get("num_steps", 8)),
            guidance_scale=float(kwargs.get("guidance_scale", 0.5)),
            cfg_normalization=bool(kwargs.get("cfg_normalization", False)),
            cfg_truncation=float(kwargs.get("cfg_truncation", 0.9)),
            max_sequence_length=int(kwargs.get("max_sequence_length", 512)),
            generator=torch.Generator(mgr.device).manual_seed(seed),
        )

        output_path = make_output_path(f"cn_{kwargs.get('control_mode', 'ctrl')}")
        result.images[0].save(output_path, quality=95)
        return output_path, seed

    # -------------------------------------------------------------------
    # Handler: Inpaint
    # -------------------------------------------------------------------
    def _run_inpaint(kwargs, task_id):
        from PIL import Image as PILImage

        seed = resolve_seed(kwargs["seed"])
        mgr.load_zit()

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
                 float(kwargs.get("control_scale", 0.9)),
                 float(kwargs.get("guidance_scale", 4.0)),
                 int(kwargs.get("num_steps", 25)), seed)

        result = pipeline(
            prompt=kwargs["prompt"],
            negative_prompt=kwargs.get("negative_prompt"),
            height=int(kwargs["height"]),
            width=int(kwargs["width"]),
            image=image,
            mask_image=mask,
            control_image=image,
            control_context_scale=float(kwargs.get("control_scale", 0.9)),
            num_inference_steps=int(kwargs.get("num_steps", 25)),
            guidance_scale=float(kwargs.get("guidance_scale", 4.0)),
            cfg_truncation=float(kwargs.get("cfg_truncation", 1.0)),
            max_sequence_length=int(kwargs.get("max_sequence_length", 512)),
            generator=torch.Generator(mgr.device).manual_seed(seed),
        )

        output_path = make_output_path("inpaint")
        result.images[0].save(output_path, quality=95)
        return output_path, seed

    # -------------------------------------------------------------------
    # Handler: Outpaint (canvas expand → inpaint)
    # -------------------------------------------------------------------
    def _run_outpaint(kwargs, task_id):
        import numpy as np
        from PIL import Image as PILImage

        seed = resolve_seed(kwargs["seed"])
        direction = kwargs.get("direction", "Right")
        dirs = direction if isinstance(direction, list) else [direction]
        expand_px = int(kwargs.get("expand_px", 256))

        log.info("Outpaint direction=%s expand=%dpx seed=%d", dirs, expand_px, seed)

        # Expand canvas
        log.info("Outpaint: preparing expanded canvas...")
        image = PILImage.open(kwargs["image_path"]).convert("RGB")
        w, h = image.size

        # Calculate new canvas
        pad = {"Left": 0, "Right": 0, "Up": 0, "Down": 0}
        for d in dirs:
            pad[d] = expand_px

        new_w = w + pad["Left"] + pad["Right"]
        new_h = h + pad["Up"] + pad["Down"]
        log.info("Outpaint: %sx%s -> %sx%s", w, h, new_w, new_h)

        # Create expanded canvas + mask
        canvas = PILImage.new("RGB", (new_w, new_h), (0, 0, 0))
        canvas.paste(image, (pad["Left"], pad["Up"]))

        mask = PILImage.new("L", (new_w, new_h), 255)  # all white = regenerate
        mask_arr = np.array(mask)
        mask_arr[pad["Up"]:pad["Up"]+h, pad["Left"]:pad["Left"]+w] = 0  # original area = preserve
        mask = PILImage.fromarray(mask_arr)

        # Save temp files and delegate to inpaint
        import tempfile
        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
        canvas.save(tmp_img.name)
        tmp_img.close()

        tmp_mask = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
        mask.save(tmp_mask.name)
        tmp_mask.close()

        # Align to 16px multiples
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16

        log.info("Outpaint: running inpaint pipeline at %sx%s...", new_w, new_h)
        inpaint_kwargs = {
            **kwargs,
            "image_path": tmp_img.name,
            "mask_path": tmp_mask.name,
            "width": new_w,
            "height": new_h,
            "seed": seed,
        }
        return _run_inpaint(inpaint_kwargs, task_id)

    # -------------------------------------------------------------------
    # Handler: FaceSwap (multi-stage: swap → restore → inpaint refine → detailer)
    # -------------------------------------------------------------------
    def _run_faceswap(kwargs, task_id):
        import numpy as np
        from PIL import Image as PILImage
        from face_swap import create_face_mask, create_detail_masks

        seed = resolve_seed(kwargs["seed"])

        mgr.load_faceswap()
        fs_pipeline = mgr.faceswap_pipeline

        target = np.array(PILImage.open(kwargs["target_path"]).convert("RGB"))
        source = np.array(PILImage.open(kwargs["source_path"]).convert("RGB"))

        det_thresh = kwargs.get("det_thresh", 0.5)
        blend_mode = kwargs.get("blend_mode", "seamless")
        mask_blur = kwargs.get("mask_blur", 0.3)
        face_index = kwargs.get("face_index", 0)
        enable_restore = kwargs.get("enable_restore", True)
        codeformer_w = kwargs.get("codeformer_w", 0.7)
        enable_refine = kwargs.get("enable_refine", True)
        refine_prompt = kwargs.get("refine_prompt", "a person with natural skin texture, highly detailed face, photorealistic")
        refine_steps = kwargs.get("refine_steps", 15)
        enable_detailer = kwargs.get("enable_detailer", False)

        log.info("FaceSwap seed=%d target=%s source=%s det=%.2f blend=%s blur=%.2f idx=%d restore=%s refine=%s detailer=%s",
                 seed, target.shape, source.shape, det_thresh, blend_mode, mask_blur, face_index,
                 enable_restore, enable_refine, enable_detailer)

        # --- Stage 1+2: Face swap + CodeFormer restoration ---
        result, swapped_faces_info = fs_pipeline.swap_face(
            target, source,
            det_thresh=det_thresh, blend_mode=blend_mode,
            mask_blur=mask_blur, face_index=face_index,
            enable_restore=enable_restore, codeformer_w=codeformer_w,
        )

        # --- Stage 3: Inpaint refinement via ZIT pipeline ---
        if enable_refine and swapped_faces_info:
            log.info("FaceSwap: running inpaint refinement (steps=%d)...", refine_steps)
            mgr.load_zit()
            zit_pipeline = mgr.zit_components["pipeline"]

            time_shift = 3.0
            zit_pipeline.scheduler = type(zit_pipeline.scheduler).from_config(
                zit_pipeline.scheduler.config, shift=time_shift,
            )

            for bbox, landmarks in swapped_faces_info:
                face_mask = create_face_mask(result.shape, bbox, landmarks, padding=1.3)
                result_pil = PILImage.fromarray(result)
                mask_pil = PILImage.fromarray(face_mask)

                h, w = result.shape[:2]
                out = zit_pipeline(
                    prompt=refine_prompt,
                    negative_prompt="blurry, low quality, artifacts, unnatural skin, wax-like",
                    height=h, width=w,
                    image=result_pil,
                    mask_image=mask_pil,
                    control_image=result_pil,
                    control_context_scale=0.95,
                    num_inference_steps=int(refine_steps),
                    guidance_scale=4.0,
                    cfg_truncation=1.0,
                    max_sequence_length=512,
                    generator=torch.Generator(mgr.device).manual_seed(seed),
                )
                result = np.array(out.images[0])

        # --- Stage 4: FaceDetailer (optional) ---
        if enable_detailer and swapped_faces_info:
            log.info("FaceSwap: running FaceDetailer...")
            if not enable_refine:
                mgr.load_zit()
                zit_pipeline = mgr.zit_components["pipeline"]
                zit_pipeline.scheduler = type(zit_pipeline.scheduler).from_config(
                    zit_pipeline.scheduler.config, shift=3.0,
                )

            detail_prompts = {
                "left_eye": "detailed realistic eye with natural iris texture, photorealistic",
                "right_eye": "detailed realistic eye with natural iris texture, photorealistic",
                "nose": "natural nose with realistic skin pores, photorealistic",
                "mouth": "natural lips and mouth with realistic skin texture, photorealistic",
            }

            for bbox, landmarks in swapped_faces_info:
                detail_parts = create_detail_masks(landmarks, result.shape)
                for part_mask, label in detail_parts:
                    if part_mask.max() == 0:
                        continue
                    result_pil = PILImage.fromarray(result)
                    mask_pil = PILImage.fromarray(part_mask)
                    prompt = detail_prompts.get(label, refine_prompt)

                    h, w = result.shape[:2]
                    out = zit_pipeline(
                        prompt=prompt,
                        negative_prompt="blurry, artifacts, unnatural",
                        height=h, width=w,
                        image=result_pil,
                        mask_image=mask_pil,
                        control_image=result_pil,
                        control_context_scale=0.95,
                        num_inference_steps=10,
                        guidance_scale=4.0,
                        cfg_truncation=1.0,
                        max_sequence_length=512,
                        generator=torch.Generator(mgr.device).manual_seed(seed),
                    )
                    result = np.array(out.images[0])

        output_path = make_output_path("faceswap")
        PILImage.fromarray(result).save(output_path, quality=95)
        return output_path, seed

    # -------------------------------------------------------------------
    # Handler dispatch
    # -------------------------------------------------------------------
    HANDLERS = {
        "zit_t2i": _run_zit_t2i,
        "controlnet": _run_controlnet,
        "inpaint": _run_inpaint,
        "outpaint": _run_outpaint,
        "faceswap": _run_faceswap,
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
