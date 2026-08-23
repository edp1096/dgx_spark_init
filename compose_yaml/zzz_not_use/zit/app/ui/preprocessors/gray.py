"""Grayscale preprocessor. No model weights needed."""

import numpy as np
from PIL import Image


def apply_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale (3-channel).

    Args:
        image: RGB numpy array (H, W, 3), uint8

    Returns:
        Grayscale as RGB (H, W, 3), uint8
    """
    gray = Image.fromarray(image).convert("L")
    return np.array(gray.convert("RGB"))
