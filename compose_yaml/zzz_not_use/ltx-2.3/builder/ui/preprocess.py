"""Video preprocessing utilities for IC-LoRA conditioning."""

import logging
from pathlib import Path

import av
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

logger = logging.getLogger("ltx2-ui")


def _canny_frame(gray: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Simple Canny-like edge detection using Sobel + double threshold."""
    smoothed = gaussian_filter(gray.astype(np.float64), sigma=1.4)
    sx = sobel(smoothed, axis=1)
    sy = sobel(smoothed, axis=0)
    magnitude = np.hypot(sx, sy)
    # Normalize to 0-255 range
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max * 255.0
    # Double threshold
    strong = magnitude >= high
    weak = (magnitude >= low) & ~strong
    # Simple hysteresis: include weak pixels adjacent to strong ones
    from scipy.ndimage import binary_dilation
    edges = strong | (weak & binary_dilation(strong, iterations=2))
    return (edges.astype(np.uint8) * 255)


def preprocess_video_canny(video_path: str, low: int = 100, high: int = 200,
                           output_dir: str = "/tmp/ltx2-outputs") -> str:
    """Apply Canny edge detection to all frames of a video. Returns output path."""
    in_container = av.open(video_path)
    in_stream = next(s for s in in_container.streams if s.type == "video")
    fps = float(in_stream.average_rate or 25)
    w = in_stream.codec_context.width
    h = in_stream.codec_context.height

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / f"_canny_{Path(video_path).stem}.mp4")
    out_container = av.open(out_path, "w", format="mp4")
    out_stream = out_container.add_stream("libx264", rate=int(fps),
                                          options={"crf": "18", "preset": "veryfast"})
    # Round to even for codec compatibility
    out_stream.height = h // 2 * 2
    out_stream.width = w // 2 * 2

    frame_count = 0
    for frame in in_container.decode(in_stream):
        rgb = frame.to_rgb().to_ndarray()
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        edges = _canny_frame(gray, low, high)
        edges_3ch = np.stack([edges, edges, edges], axis=-1)
        # Crop to even dimensions
        edges_3ch = edges_3ch[:out_stream.height, :out_stream.width]
        av_frame = av.VideoFrame.from_ndarray(edges_3ch, format="rgb24")
        out_container.mux(out_stream.encode(av_frame))
        frame_count += 1

    out_container.mux(out_stream.encode())
    in_container.close()
    out_container.close()
    logger.info("Canny preprocessed %d frames → %s", frame_count, out_path)
    return out_path


def preview_canny(video_path: str, low: int = 100, high: int = 200,
                   num_frames: int = 3) -> list[tuple[np.ndarray, str]] | None:
    """Extract a few sample frames, apply Canny, return as (image, caption) list."""
    try:
        container = av.open(video_path)
        stream = next(s for s in container.streams if s.type == "video")
        total = stream.frames or 0
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        # Decode all frame indices if total unknown
        if total < num_frames:
            # Short video: just grab everything
            indices = set(range(max(total, 999)))
        else:
            # Evenly spaced: first, middle, last
            indices = set(int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames))

        results = []
        for idx, frame in enumerate(container.decode(stream)):
            if idx in indices:
                rgb = frame.to_rgb().to_ndarray()
                gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                edges = _canny_frame(gray, low, high)
                img = np.stack([edges, edges, edges], axis=-1)
                caption = f"Frame {idx} ({idx / fps:.2f}s)"
                results.append((img, caption))
                if len(results) >= num_frames:
                    break
        container.close()
        logger.info("Canny preview: %d sample frames from %s", len(results), video_path)
        return results if results else None
    except Exception:
        logger.exception("Canny preview failed")
        return None
