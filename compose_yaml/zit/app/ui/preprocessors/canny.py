"""Canny edge detection preprocessor. No model weights needed."""

import cv2
import numpy as np


def apply_canny(
    image: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> np.ndarray:
    """Extract Canny edges from RGB image.

    Args:
        image: RGB numpy array (H, W, 3), uint8
        low_threshold: Canny low threshold
        high_threshold: Canny high threshold

    Returns:
        RGB edge map (H, W, 3), uint8
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
