"""Test if worker process can spawn and import pipeline_manager."""
import sys
import os
import multiprocessing as mp

sys.path.insert(0, "app/ui")

def test_worker(result_queue):
    """Minimal worker — just import and report."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("test-worker")
    try:
        log.info("Worker PID=%d started", os.getpid())
        import torch
        log.info("torch ok, cuda=%s", torch.cuda.is_available())
        from pipeline_manager import PipelineManager
        log.info("PipelineManager imported ok")
        mgr = PipelineManager()
        log.info("PipelineManager created, device=%s", mgr.device)
        result_queue.put("OK")
    except Exception as e:
        import traceback
        log.error("FAIL: %s\n%s", e, traceback.format_exc())
        result_queue.put(f"FAIL: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=test_worker, args=(q,), name="test-zit-worker")
    print(f"Starting worker process...")
    p.start()
    p.join(timeout=30)
    if p.is_alive():
        print("TIMEOUT — worker hung during import!")
        p.kill()
        p.join()
        sys.exit(1)
    else:
        try:
            result = q.get_nowait()
            print(f"Worker result: {result}")
            sys.exit(0 if result == "OK" else 1)
        except Exception:
            print(f"Worker exited with code {p.exitcode}, no result")
            sys.exit(1)
