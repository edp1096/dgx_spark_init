"""HED (Holistically-Nested Edge Detection) preprocessor.

Uses ControlNetHED_Apache2 model from lllyasviel/Annotators (Apache 2.0 license).
Model weights: ControlNetHED.pth (~29 MB)
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = nn.Sequential()
        self.convs.append(nn.Conv2d(input_channel, output_channel, 3, 1, 1))
        for _ in range(1, layer_number):
            self.convs.append(nn.Conv2d(output_channel, output_channel, 3, 1, 1))
        self.projection = nn.Conv2d(output_channel, 1, 1, 1, 0)

    def forward(self, x, down_sampling=False):
        if down_sampling:
            x = F.max_pool2d(x, 2, 2)
        for conv in self.convs:
            x = F.relu(conv(x))
        return x, self.projection(x)


class ControlNetHED(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.block1 = DoubleConvBlock(3, 64, 2)
        self.block2 = DoubleConvBlock(64, 128, 2)
        self.block3 = DoubleConvBlock(128, 256, 3)
        self.block4 = DoubleConvBlock(256, 512, 3)
        self.block5 = DoubleConvBlock(512, 512, 3)

    def forward(self, x):
        h = x - self.norm
        h, p1 = self.block1(h)
        h, p2 = self.block2(h, down_sampling=True)
        h, p3 = self.block3(h, down_sampling=True)
        h, p4 = self.block4(h, down_sampling=True)
        h, p5 = self.block5(h, down_sampling=True)
        return p1, p2, p3, p4, p5


# Singleton model cache
_hed_model = None


def _load_hed_model(model_dir: str) -> ControlNetHED:
    global _hed_model
    if _hed_model is not None:
        return _hed_model

    from zit_config import PREPROCESSORS_DIR, HED_FILE
    hed_path = os.path.join(model_dir, PREPROCESSORS_DIR, HED_FILE)
    if not os.path.exists(hed_path):
        raise FileNotFoundError(f"HED model not found: {hed_path}")

    model = ControlNetHED()
    model.load_state_dict(torch.load(hed_path, map_location="cpu"))
    model = model.cuda().eval()
    _hed_model = model
    return model


def apply_hed(image: np.ndarray, model_dir: str) -> np.ndarray:
    """Apply HED edge detection.

    Args:
        image: RGB numpy array (H, W, 3), uint8
        model_dir: Path to model directory

    Returns:
        Soft edge map as RGB (H, W, 3), uint8
    """
    model = _load_hed_model(model_dir)
    H, W = image.shape[:2]

    with torch.no_grad():
        x = torch.from_numpy(image.copy()).float().cuda()
        x = x.permute(2, 0, 1).unsqueeze(0)  # HWC → 1CHW
        edges = model(x)
        edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
        edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
        edges = np.stack(edges, axis=2)
        edge = 1.0 / (1.0 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

    return cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)


def _nms(x: np.ndarray, t: float = 128, s: float = 3.0) -> np.ndarray:
    """Non-maximum suppression for thin edge lines."""
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)
    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    y = np.zeros_like(x)
    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)
    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


def apply_scribble(image: np.ndarray, model_dir: str) -> np.ndarray:
    """Apply HED + NMS for scribble-style edges.

    Args:
        image: RGB numpy array (H, W, 3), uint8
        model_dir: Path to model directory

    Returns:
        Scribble edge map as RGB (H, W, 3), uint8
    """
    hed_result = apply_hed(image, model_dir)
    # Convert to grayscale for NMS
    gray = cv2.cvtColor(hed_result, cv2.COLOR_RGB2GRAY)
    scribble = _nms(gray)
    return cv2.cvtColor(scribble, cv2.COLOR_GRAY2RGB)
