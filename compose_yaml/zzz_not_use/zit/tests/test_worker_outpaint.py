"""Test worker outpaint flow — spawn worker, submit outpaint task, watch result.

Simulates exactly what Gradio does: spawn worker, submit task via queue, wait for result.
"""
import sys
import os
import time
import tempfile
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "ui"))

import numpy as np
from PIL import Image


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("test")

    MODEL_DIR = "/root/.cache/huggingface/hub/zit"
    OUTPUT_DIR = Path("/tmp/zit-outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create test image
    log.info("Creating test image...")
    img_arr = np.full((256, 256, 3), (100, 150, 200), dtype=np.uint8)
    img = Image.fromarray(img_arr)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
    img.save(tmp.name)
    tmp.close()
    log.info("Test image: %s", tmp.name)

    # Import worker
    from worker import _worker_loop

    # Create queues
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    progress_q = ctx.Queue()

    # Start worker
    log.info("Starting worker process...")
    proc = ctx.Process(
        target=_worker_loop,
        args=(task_q, result_q, progress_q, MODEL_DIR),
        daemon=True,
        name="test-zit-worker",
    )
    proc.start()
    log.info("Worker PID=%d started", proc.pid)

    # Submit outpaint task
    task_id = "test-outpaint-001"
    task = {
        "task_id": task_id,
        "gen_type": "outpaint",
        "kwargs": {
            "prompt": "beautiful landscape extending to the right",
            "negative_prompt": None,
            "image_path": tmp.name,
            "direction": ["Right"],
            "expand_px": 128,
            "control_scale": 0.9,
            "num_steps": 4,
            "guidance_scale": 0.5,
            "cfg_truncation": 0.9,
            "max_sequence_length": 512,
            "time_shift": 3.0,
            "seed": 42,
        },
    }
    log.info("Submitting outpaint task: %s", task_id)
    task_q.put(task)

    # Poll for result
    start = time.time()
    timeout = 300  # 5 min max
    last_progress = time.time()

    while time.time() - start < timeout:
        # Check worker alive
        if not proc.is_alive():
            log.error("WORKER CRASHED! exitcode=%s", proc.exitcode)
            break

        # Poll progress
        while True:
            try:
                msg = progress_q.get_nowait()
                elapsed = time.time() - start
                log.info("[%.1fs] Progress: %s — %s", elapsed, msg.get("type"), msg.get("data"))
                last_progress = time.time()
            except Exception:
                break

        # Check result
        try:
            result = result_q.get(timeout=1.0)
            elapsed = time.time() - start
            log.info("[%.1fs] Got result!", elapsed)
            log.info("  status: %s", result.get("status"))
            if result["status"] == "ok":
                payload = result["payload"]
                log.info("  paths: %s", payload.get("paths"))
                log.info("  seed: %s", payload.get("seed"))
                log.info("SUCCESS — outpaint completed in %.1fs", elapsed)
            else:
                log.error("  error: %s", result.get("payload"))
                log.error("FAILED — outpaint error")

            # Shutdown
            task_q.put(None)
            proc.join(timeout=5)
            os.unlink(tmp.name)
            return 0 if result["status"] == "ok" else 1

        except Exception:
            pass

        # Heartbeat every 15s
        if time.time() - last_progress > 15:
            elapsed = time.time() - start
            log.info("[%.1fs] Still waiting... worker alive=%s", elapsed, proc.is_alive())
            last_progress = time.time()

    log.error("TIMEOUT after %ds — worker alive=%s", timeout, proc.is_alive())
    if proc.is_alive():
        proc.kill()
    os.unlink(tmp.name)
    return 1


if __name__ == "__main__":
    sys.exit(main())
