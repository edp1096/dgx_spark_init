"""Video preprocessing utilities for IC-LoRA conditioning."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("ltx2-ui")


def preprocess_video_canny(video_path: str, low: int = 100, high: int = 200,
                           output_dir: str = "/tmp/ltx2-outputs") -> str:
    """Apply Canny edge detection to all frames of a video. Returns output path."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = str(Path(output_dir) / f"_canny_{Path(video_path).stem}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        writer.write(edges_3ch)
        frame_count += 1

    cap.release()
    writer.release()
    logger.info("Canny preprocessed %d frames → %s", frame_count, out_path)
    return out_path


def preview_canny(video_path: str, low: int = 100, high: int = 200) -> np.ndarray | None:
    """Extract first frame and apply Canny for preview. Returns RGB numpy array."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
