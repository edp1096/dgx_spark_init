"""DWPose preprocessor for ZIT ControlNet pose mode.

Loads YOLOX (person detection) + DWPose (keypoint estimation) ONNX models
via cv2.dnn backend and renders an OpenPose-style skeleton map.

Model files are loaded from ``model_dir/preprocessors/``:
  - yolox_l.onnx       (DWPOSE_DET_FILE)
  - dw-ll_ucoco_384.onnx (DWPOSE_POSE_FILE)
"""

import os
from typing import Optional

import numpy as np

from .dwpose_utils import DWposeDetector

# Singleton detector -- loaded once, reused across calls
_detector: Optional[DWposeDetector] = None


def apply_dwpose(image: np.ndarray, model_dir: str) -> np.ndarray:
    """Apply DWPose estimation and render skeleton on black canvas.

    Args:
        image: RGB numpy array (H, W, 3), uint8
        model_dir: Root model directory (contains preprocessors/ subdir)

    Returns:
        RGB pose map (H, W, 3), uint8 -- skeleton on black background
    """
    global _detector

    if _detector is None:
        from zit_config import PREPROCESSORS_DIR, DWPOSE_DET_FILE, DWPOSE_POSE_FILE

        prep_dir = os.path.join(model_dir, PREPROCESSORS_DIR)
        det_path = os.path.join(prep_dir, DWPOSE_DET_FILE)
        pose_path = os.path.join(prep_dir, DWPOSE_POSE_FILE)

        if not os.path.exists(det_path):
            raise FileNotFoundError(f"DWPose detection model not found: {det_path}")
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"DWPose pose model not found: {pose_path}")

        _detector = DWposeDetector(det_path, pose_path)

    return _detector(image)
