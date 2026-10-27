"""One durable job, isolated process group. Existing successful stages are reused."""
import json
import math
import os
import subprocess
import sys
import time

import httpx

import store

TALK = os.environ.get("SHORTS_TALK_ENDPOINT", "http://127.0.0.1:8585").rstrip("/")


def bridge(operation, payload):
    with httpx.Client(timeout=1200) as client:
        response = client.post(TALK + "/api/shorts/" + operation, json=payload)
        if response.status_code != 200:
            raise RuntimeError(response.text[:2000])
        return response.json()


def render_frames(folder, job_id):
    arguments = ["hyperframes", "render", str(folder), "--output", str(folder / "render.mp4"), "--fps", "24", "--workers", "1", "--no-browser-gpu", "--low-memory-mode", "--no-best-effort"]
    process = subprocess.Popen(arguments, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    capture_started, last_update = None, 0
    for line in process.stdout:
        print(line, end="", flush=True)
        if not line.startswith("@hf-progress "):
            continue
        data = json.loads(line[len("@hf-progress "):])
        now = time.monotonic()
        if data.get("code") != "capture" or data.get("done", 0) < 1:
            continue
        if capture_started is None:
            capture_started = now
        done, total = data["done"], data.get("total", 0)
        if total and (now - last_update >= 1 or done >= total):
            eta = (now - capture_started) / done * max(0, total - done)
            label = f"세로 영상 렌더링 {done}/{total}프레임"
            if done >= 12 and eta > 1:
                label += f" · 약 {math.ceil(eta)}초 남음"
            store.update(job_id, "running", label, 75 + 17 * done / total)
            last_update = now
    if process.wait():
        raise RuntimeError("HyperFrames 영상 렌더링에 실패했습니다. 작업 로그를 보존했습니다.")


def run(job_id):
    from pipeline import produce
    produce(job_id, bridge, render_frames)


if __name__ == "__main__":
    try:
        run(sys.argv[1])
    except Exception as error:
        (store.ROOT / sys.argv[1] / "error.txt").write_text(str(error))
        raise
