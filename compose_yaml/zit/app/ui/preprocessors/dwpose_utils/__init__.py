"""DWPose wholebody pose estimation utilities.

Adapted from VideoX-Fun/comfyui/annotator/dwpose_utils/
Original: https://github.com/Mikubill/sd-webui-controlnet (Openpose/CMU)

This version uses cv2.dnn backend exclusively (no onnxruntime dependency).
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from . import util
from .wholebody import Wholebody


def draw_pose(poses, H, W):
    """Draw all detected poses on a blank canvas.

    Args:
        poses: List of HumanPoseResult from Wholebody.format_result().
        H: Canvas height.
        W: Canvas width.

    Returns:
        numpy array (H, W, 3) with drawn pose skeleton.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        canvas = util.draw_bodypose(canvas, pose.body.keypoints)
        canvas = util.draw_handpose(canvas, pose.left_hand)
        canvas = util.draw_handpose(canvas, pose.right_hand)
        canvas = util.draw_facepose(canvas, pose.face)
    return canvas


class DWposeDetector:
    """DWPose detector using cv2.dnn ONNX backend.

    Args:
        onnx_det: Path to YOLOX detection ONNX model.
        onnx_pose: Path to DWPose keypoint ONNX model.
    """

    def __init__(self, onnx_det: str, onnx_pose: str):
        self.pose_estimation = Wholebody(onnx_det, onnx_pose)

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        keypoints_info = self.pose_estimation(oriImg)
        return draw_pose(
            Wholebody.format_result(keypoints_info),
            H,
            W,
        )
