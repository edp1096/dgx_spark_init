"""ZoeDepth preprocessor for depth map generation.

Adapted from VideoX-Fun (comfyui/annotator/zoe/ + comfyui/annotator/nodes.py).
Model weights: ZoeD_M12_N.pt (~350 MB)
License: MIT (see zoe/LICENSE)
"""

import os

import cv2
import numpy as np
import torch

# Singleton model cache
_model = None


def _pad64(x: int) -> int:
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def _safer_memory(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.copy()).copy()


def _HWC3(x: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel HWC uint8."""
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def _resize_image_with_pad(input_image: np.ndarray, resolution: int):
    """Resize image so that the shorter side equals resolution, then pad to
    multiples of 64. Returns (padded_image, remove_pad_fn).

    Ported from VideoX-Fun comfyui/annotator/nodes.py.
    """
    img = _HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = _pad64(H_target), _pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return _safer_memory(x[:H_target, :W_target])

    return _safer_memory(img_padded), remove_pad


def _load_model(model_dir: str):
    """Build and load ZoeDepth model (singleton)."""
    global _model
    if _model is not None:
        return _model

    from .zoe import ZoeDepth, get_config
    from zit_config import PREPROCESSORS_DIR, ZOEDEPTH_FILE

    zoe_path = os.path.join(model_dir, PREPROCESSORS_DIR, ZOEDEPTH_FILE)
    if not os.path.exists(zoe_path):
        raise FileNotFoundError(f"ZoeDepth model not found: {zoe_path}")

    # Build model from config (infer mode, no pretrained backbone download)
    conf = get_config("zoedepth", "infer")
    model = ZoeDepth.build_from_config(conf)

    # Load checkpoint weights
    model.load_state_dict(
        torch.load(zoe_path, map_location="cpu")['model'],
        strict=False,
    )
    model = model.cuda().eval().requires_grad_(False)
    _model = model
    return model


def apply_zoedepth(image: np.ndarray, model_dir: str) -> np.ndarray:
    """Apply ZoeDepth to produce a depth map.

    Pipeline (from VideoX-Fun ImageToDepth.process_single_image):
      1. resize_image_with_pad to 512 (shorter side)
      2. Normalize to [0, 1], rearrange HWC -> 1CHW
      3. model.infer() -> depth tensor
      4. Percentile normalization (2nd/85th), inversion, uint8
      5. Remove padding, ensure 3-channel output

    Args:
        image: RGB numpy array (H, W, 3), uint8
        model_dir: Path to model directory

    Returns:
        Depth map as RGB (H, W, 3), uint8 -- inverted (closer = brighter)
    """
    model = _load_model(model_dir)

    with torch.no_grad():
        # Step 1: Resize with pad to 512
        image_padded, remove_pad = _resize_image_with_pad(image, 512)

        # Step 2: Convert to tensor, normalize, rearrange HWC -> 1CHW
        image_t = torch.from_numpy(image_padded).float().cuda()
        image_t = image_t / 255.0
        image_t = image_t.permute(2, 0, 1).unsqueeze(0)  # HWC -> 1CHW

        # Step 3: Infer depth
        depth = model.infer(image_t)
        depth = depth[0, 0].cpu().numpy()

        # Step 4: Percentile normalization + inversion
        vmin = np.percentile(depth, 2)
        vmax = np.percentile(depth, 85)
        depth -= vmin
        depth /= (vmax - vmin + 1e-8)
        depth = 1.0 - depth
        depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

        # Step 5: Remove padding + ensure 3-channel
        depth_image = remove_pad(depth_image)
        depth_image = _HWC3(depth_image)

    return depth_image
