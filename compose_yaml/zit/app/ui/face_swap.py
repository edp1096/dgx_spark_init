"""Face detection + auto-mask generation for ZIT Inpaint-based face editing.

Approach (same as ComfyUI Z-Image Turbo Headswap):
  1. SCRFD detects face bbox + 5-point landmarks (via cv2.dnn, no onnxruntime)
  2. Generate soft elliptical mask from bbox/landmarks
  3. Pass mask to ZIT Inpaint pipeline → regenerate face region with prompt

No inswapper, ArcFace, or CodeFormer needed.
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger("zit-ui")


# ---------------------------------------------------------------------------
# SCRFD face detection via cv2.dnn
# ---------------------------------------------------------------------------
class SCRFDDetector:
    """Face detection using SCRFD via cv2.dnn (proven on aarch64/Blackwell)."""

    def __init__(self, onnx_path: str):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"SCRFD model not found: {onnx_path}")
        logger.info("Loading SCRFD via cv2.dnn: %s", onnx_path)
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._output_names = self.net.getUnconnectedOutLayersNames()
        logger.info("SCRFD loaded (%d outputs)", len(self._output_names))

    def detect(self, image: np.ndarray, threshold: float = 0.5, det_size: tuple = (320, 320)):
        """Detect faces in RGB image.

        Returns:
            list of (bbox[4], landmarks[5,2]) sorted by face area (largest first)
        """
        h, w = image.shape[:2]

        # Resize keeping aspect ratio, zero-pad to det_size
        im_ratio = float(h) / w
        model_ratio = float(det_size[1]) / det_size[0]
        if im_ratio > model_ratio:
            new_height = det_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = det_size[0]
            new_height = int(new_width * im_ratio)
        scale = float(new_height) / h

        det_img = np.zeros((det_size[1], det_size[0], 3), dtype=np.uint8)
        resized = cv2.resize(image, (new_width, new_height))
        det_img[:new_height, :new_width, :] = resized

        # SCRFD expects RGB, normalized (x - 127.5) / 128
        # Input is already RGB; blobFromImage with swapRB=False keeps it as-is
        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0, det_size,
            (127.5, 127.5, 127.5), swapRB=False,
        )

        # Run
        self.net.setInput(blob)
        raw_outputs = self.net.forward(list(self._output_names))

        # Parse SCRFD outputs: 9 tensors grouped by stride
        # (score_s8, bbox_s8, kps_s8, score_s16, bbox_s16, kps_s16, score_s32, bbox_s32, kps_s32)
        out_sorted = sorted(zip(self._output_names, raw_outputs), key=lambda x: x[0])
        outputs = [o for _, o in out_sorted]

        num_anchors = 2
        strides = [8, 16, 32]

        all_scores, all_bboxes, all_kps = [], [], []

        for idx, stride in enumerate(strides):
            scores = outputs[idx * 3 + 0].reshape(-1)
            bbox_preds = outputs[idx * 3 + 1].reshape(-1, 4)
            kps_preds = outputs[idx * 3 + 2].reshape(-1, 10)

            fh = det_size[1] // stride
            fw = det_size[0] // stride

            anchor_centers = np.stack(
                np.mgrid[:fh, :fw][::-1], axis=-1
            ).astype(np.float32).reshape(-1, 2) * stride

            if num_anchors > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * num_anchors, axis=1
                ).reshape(-1, 2)

            mask = scores > threshold
            if not mask.any():
                continue

            scores = scores[mask]
            bbox_preds = bbox_preds[mask]
            kps_preds = kps_preds[mask]
            centers = anchor_centers[mask]

            bboxes = np.column_stack([
                centers[:, 0] - bbox_preds[:, 0] * stride,
                centers[:, 1] - bbox_preds[:, 1] * stride,
                centers[:, 0] + bbox_preds[:, 2] * stride,
                centers[:, 1] + bbox_preds[:, 3] * stride,
            ])

            kps = np.zeros_like(kps_preds)
            for k in range(kps_preds.shape[1]):
                kps[:, k] = centers[:, k % 2] + kps_preds[:, k] * stride

            all_scores.append(scores)
            all_bboxes.append(bboxes)
            all_kps.append(kps)

        if not all_scores:
            return []

        all_scores = np.concatenate(all_scores)
        all_bboxes = np.concatenate(all_bboxes)
        all_kps = np.concatenate(all_kps)

        # NMS
        keep = self._nms(all_bboxes, all_scores, iou_threshold=0.4)
        all_bboxes = all_bboxes[keep]
        all_kps = all_kps[keep]
        all_scores = all_scores[keep]

        # Scale back + sort by area (largest first)
        faces = []
        for bbox, kps in zip(all_bboxes, all_kps):
            bbox = bbox / scale
            kps = kps.reshape(5, 2) / scale
            faces.append((bbox, kps))

        faces.sort(key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]), reverse=True)
        return faces

    @staticmethod
    def _nms(bboxes, scores, iou_threshold=0.4):
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        return keep


# ---------------------------------------------------------------------------
# Singleton detector
# ---------------------------------------------------------------------------
_detector: SCRFDDetector | None = None


def get_detector(model_dir: str) -> SCRFDDetector:
    global _detector
    if _detector is None:
        from zit_config import PREPROCESSORS_DIR, SCRFD_FILE
        scrfd_path = os.path.join(model_dir, PREPROCESSORS_DIR, SCRFD_FILE)
        _detector = SCRFDDetector(scrfd_path)
    return _detector


# ---------------------------------------------------------------------------
# Auto-mask generation from face detection
# ---------------------------------------------------------------------------
def create_face_mask(
    image: np.ndarray,
    model_dir: str,
    face_index: int = 0,
    padding: float = 1.3,
    det_threshold: float = 0.5,
) -> tuple[np.ndarray | None, list]:
    """Detect faces and create soft elliptical mask for the selected face.

    Args:
        image: RGB numpy array (H, W, 3)
        model_dir: path to model directory
        face_index: which face to mask (0=largest, -1=all)
        padding: bbox expansion factor (1.0=tight, 1.5=loose)
        det_threshold: SCRFD detection confidence

    Returns:
        (mask, faces) — mask is L-mode uint8 (0=keep, 255=inpaint), None if no face found
        faces is list of (bbox, landmarks) for all detected faces
    """
    detector = get_detector(model_dir)
    faces = detector.detect(image, threshold=det_threshold)

    if not faces:
        return None, []

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if face_index >= 0:
        targets = [faces[min(face_index, len(faces) - 1)]]
    else:
        targets = faces

    for bbox, landmarks in targets:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Expand bbox
        bw *= padding
        bh *= padding
        ex1 = max(0, int(cx - bw / 2))
        ey1 = max(0, int(cy - bh / 2))
        ex2 = min(w, int(cx + bw / 2))
        ey2 = min(h, int(cy + bh / 2))

        # Draw filled ellipse
        ecx = (ex1 + ex2) // 2
        ecy = (ey1 + ey2) // 2
        eax = (ex2 - ex1) // 2
        eay = (ey2 - ey1) // 2
        cv2.ellipse(mask, (ecx, ecy), (eax, eay), 0, 0, 360, 255, -1)

    # Feather edges
    ksize = max(3, int(min(eax, eay) * 0.25) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    return mask, faces


def preview_face_detection(image: np.ndarray, model_dir: str, det_threshold: float = 0.5) -> np.ndarray:
    """Draw face detection results on image for preview.

    Returns RGB image with bboxes, landmarks, and mask overlay drawn.
    """
    detector = get_detector(model_dir)
    faces = detector.detect(image, threshold=det_threshold)

    preview = image.copy()
    for bbox, landmarks in faces:
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for lm in landmarks:
            cv2.circle(preview, (int(lm[0]), int(lm[1])), 3, (255, 0, 0), -1)

    # Overlay mask
    mask, _ = create_face_mask(image, model_dir, face_index=-1, det_threshold=det_threshold)
    if mask is not None:
        overlay = preview.copy()
        overlay[mask > 128] = (overlay[mask > 128] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        preview = overlay

    return preview
