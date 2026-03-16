"""TensorRT-based FaceSwap pipeline for ZIT.

Replaces InsightFace + onnxruntime with pure TensorRT inference.
Components:
  - SCRFD: Face detection (bounding boxes + 5-point landmarks)
  - ArcFace: Face embedding extraction (512-dim)
  - inswapper: Face replacement (128x128 aligned face + source embedding → swapped face)

ONNX → TRT engine conversion is done on first use and cached as .engine files.
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("zit-ui")


# ---------------------------------------------------------------------------
# TensorRT engine wrapper
# ---------------------------------------------------------------------------
class TRTEngine:
    """Generic TensorRT engine: ONNX → TRT build + cached inference."""

    def __init__(self, onnx_path: str, engine_path: str = None, fp16: bool = True):
        import tensorrt as trt
        import torch

        self.logger = trt.Logger(trt.Logger.WARNING)

        if engine_path is None:
            engine_path = onnx_path.replace(".onnx", ".engine")
        self.engine_path = engine_path

        # Build engine if not cached
        if not os.path.exists(engine_path):
            logger.info("Building TRT engine: %s → %s (this may take 30-60s)", onnx_path, engine_path)
            self._build_engine(onnx_path, engine_path, fp16)

        # Load engine
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Discover I/O bindings
        self.inputs = {}
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)
            info = {"name": name, "shape": list(shape), "dtype": dtype}
            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = info
            else:
                self.outputs[name] = info

    def _build_engine(self, onnx_path: str, engine_path: str, fp16: bool):
        import tensorrt as trt

        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("TRT parse error: %s", parser.get_error(i))
                raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8GB
        config.clear_flag(trt.BuilderFlag.TF32)  # GB10 compatibility
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Add optimization profile for dynamic input shapes
        has_dynamic = False
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = inp.shape
            if -1 in shape:
                has_dynamic = True
                break

        if has_dynamic:
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                inp = network.get_input(i)
                shape = list(inp.shape)
                # Replace dynamic dims: batch → 1, spatial → 640
                concrete = []
                for j, s in enumerate(shape):
                    if s != -1:
                        concrete.append(s)
                    elif j == 0:
                        concrete.append(1)  # batch dim
                    else:
                        concrete.append(640)  # spatial dim
                profile.set_shape(inp.name, concrete, concrete, concrete)
            config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError(f"Failed to build TRT engine for {onnx_path}")

        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        logger.info("TRT engine saved: %s", engine_path)

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with numpy arrays."""
        import torch

        # Keep input tensors alive during inference
        input_tensors = {}
        for name, arr in inputs.items():
            t = torch.from_numpy(arr).cuda()
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
            input_tensors[name] = t

        # Allocate output buffers using resolved shapes from context
        output_buffers = {}
        for name in self.outputs:
            shape = self.context.get_tensor_shape(name)
            output_buffers[name] = torch.empty(
                tuple(shape), dtype=torch.float32, device="cuda"
            )
            self.context.set_tensor_address(name, output_buffers[name].data_ptr())

        # Execute
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        return {name: buf.cpu().numpy() for name, buf in output_buffers.items()}


# ---------------------------------------------------------------------------
# Face alignment utilities (pure numpy/cv2)
# ---------------------------------------------------------------------------
# Standard 5-point reference for 112x112 aligned face (ArcFace standard)
ARCFACE_REF_5 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> tuple:
    """Align face using 5-point landmarks → affine transform.

    Returns:
        aligned: Aligned face crop (size x size x 3)
        M: Affine transformation matrix (for inverse mapping)
    """
    ref = ARCFACE_REF_5.copy()
    if size != 112:
        ref = ref * (size / 112.0)

    M = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ref)[0]
    aligned = cv2.warpAffine(image, M, (size, size), borderValue=(0, 0, 0))
    return aligned, M


def paste_back(swapped_face: np.ndarray, target: np.ndarray, M: np.ndarray, size: int = 128) -> np.ndarray:
    """Paste swapped face back onto target image using inverse affine transform."""
    M_inv = cv2.invertAffineTransform(M)
    face_back = cv2.warpAffine(swapped_face, M_inv, (target.shape[1], target.shape[0]))

    # Create mask from warped face
    mask = np.zeros(swapped_face.shape[:2], dtype=np.float32)
    # Feathered mask (elliptical, softer edges)
    center = (size // 2, size // 2)
    axes = (int(size * 0.4), int(size * 0.45))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), size * 0.05)

    mask_back = cv2.warpAffine(mask, M_inv, (target.shape[1], target.shape[0]))
    mask_back = mask_back[:, :, np.newaxis]

    result = target.astype(np.float32) * (1 - mask_back) + face_back.astype(np.float32) * mask_back
    return result.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# FaceSwap Pipeline
# ---------------------------------------------------------------------------
class FaceSwapPipeline:
    """Complete face swap: detect → align → recognize → swap → paste back."""

    def __init__(self, model_dir: str):
        from zit_config import FACESWAP_DIR, SCRFD_FILE, ARCFACE_FILE, INSWAPPER_FILE

        self.model_dir = model_dir
        fs_dir = os.path.join(model_dir, FACESWAP_DIR)

        self.scrfd_path = os.path.join(fs_dir, SCRFD_FILE)
        self.arcface_path = os.path.join(fs_dir, ARCFACE_FILE)
        self.inswapper_path = os.path.join(fs_dir, INSWAPPER_FILE)

        self._scrfd = None
        self._arcface = None
        self._inswapper = None
        self._emap = None

    def _ensure_loaded(self):
        if self._scrfd is not None:
            return

        for path in [self.scrfd_path, self.arcface_path, self.inswapper_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"FaceSwap model not found: {path}")

        logger.info("Loading FaceSwap TRT engines...")
        self._scrfd = TRTEngine(self.scrfd_path, fp16=False)
        self._arcface = TRTEngine(self.arcface_path, fp16=False)
        self._inswapper = TRTEngine(self.inswapper_path, fp16=False)

        # Load emap matrix from inswapper ONNX (embedding transform)
        import onnx
        from onnx import numpy_helper
        onnx_model = onnx.load(self.inswapper_path)
        self._emap = numpy_helper.to_array(onnx_model.graph.initializer[-1])
        del onnx_model

        logger.info("FaceSwap engines loaded")

    def detect_faces(self, image: np.ndarray, threshold: float = 0.5):
        """Detect faces using SCRFD. Returns list of (bbox, landmarks_5)."""
        self._ensure_loaded()

        # Preprocess: resize keeping aspect ratio, zero-pad to 640x640
        h, w = image.shape[:2]
        det_size = (640, 640)
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

        # BGR input for blobFromImage (swapRB=True converts to RGB)
        det_img_bgr = cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(
            det_img_bgr, 1.0 / 128.0, det_size,
            (127.5, 127.5, 127.5), swapRB=True,
        )

        # Run SCRFD
        input_name = list(self._scrfd.inputs.keys())[0]
        raw = self._scrfd.infer({input_name: blob})

        # Parse SCRFD 10g_bnkps outputs: 9 tensors, sorted by name →
        # stride-grouped: (score_s8, bbox_s8, kps_s8, score_s16, bbox_s16, kps_s16, score_s32, bbox_s32, kps_s32)
        out_names = sorted(raw.keys())
        num_anchors = 2  # scrfd_10g uses 2 anchors per position
        strides = [8, 16, 32]

        all_scores = []
        all_bboxes = []
        all_kps = []

        for idx, stride in enumerate(strides):
            scores = raw[out_names[idx * 3 + 0]].reshape(-1)
            bbox_preds = raw[out_names[idx * 3 + 1]].reshape(-1, 4)
            kps_preds = raw[out_names[idx * 3 + 2]].reshape(-1, 10)

            fh = det_size[1] // stride
            fw = det_size[0] // stride

            # Generate anchor centers (insightface style)
            anchor_centers = np.stack(
                np.mgrid[:fh, :fw][::-1], axis=-1
            ).astype(np.float32).reshape(-1, 2)
            anchor_centers = (anchor_centers * stride)
            if num_anchors > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * num_anchors, axis=1
                ).reshape(-1, 2)

            # Filter by threshold
            mask = scores > threshold
            if not mask.any():
                continue

            scores = scores[mask]
            bbox_preds = bbox_preds[mask]
            kps_preds = kps_preds[mask]
            centers = anchor_centers[mask]

            # Decode bboxes: distance2bbox (left, top, right, bottom) × stride
            bboxes = np.column_stack([
                centers[:, 0] - bbox_preds[:, 0] * stride,
                centers[:, 1] - bbox_preds[:, 1] * stride,
                centers[:, 0] + bbox_preds[:, 2] * stride,
                centers[:, 1] + bbox_preds[:, 3] * stride,
            ])

            # Decode landmarks: distance2kps × stride
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
        order = all_scores.argsort()[::-1]
        all_scores = all_scores[order]
        all_bboxes = all_bboxes[order]
        all_kps = all_kps[order]

        keep = self._nms(all_bboxes, all_scores, iou_threshold=0.4)
        all_bboxes = all_bboxes[keep]
        all_kps = all_kps[keep]

        # Scale back to original image coords
        faces = []
        for bbox, kps in zip(all_bboxes, all_kps):
            bbox = bbox / scale
            kps = kps.reshape(5, 2) / scale
            faces.append((bbox, kps))

        return faces

    @staticmethod
    def _nms(bboxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4) -> list[int]:
        """Non-maximum suppression."""
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
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Extract face embedding using ArcFace."""
        self._ensure_loaded()

        aligned, _ = align_face(image, landmarks, size=112)
        blob = aligned.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        blob = (blob - 127.5) / 127.5  # Normalize to [-1, 1]

        input_name = list(self._arcface.inputs.keys())[0]
        outputs = self._arcface.infer({input_name: blob})
        embedding = list(outputs.values())[0].flatten()
        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding

    def swap_face(
        self,
        target_image: np.ndarray,
        source_image: np.ndarray,
    ) -> np.ndarray:
        """Swap face from source onto target.

        Args:
            target_image: RGB image with face to replace
            source_image: RGB image with source face

        Returns:
            RGB image with swapped face
        """
        self._ensure_loaded()

        # Detect faces
        target_faces = self.detect_faces(target_image)
        source_faces = self.detect_faces(source_image)

        if not target_faces:
            raise ValueError("No face detected in target image")
        if not source_faces:
            raise ValueError("No face detected in source image")

        # Get source embedding and transform via emap
        src_bbox, src_landmarks = source_faces[0]
        source_embedding = self.get_embedding(source_image, src_landmarks)
        latent = np.dot(source_embedding.reshape(1, -1), self._emap)
        latent /= (np.linalg.norm(latent) + 1e-8)

        # Swap each target face
        result = target_image.copy()
        for bbox, landmarks in target_faces:
            aligned, M = align_face(result, landmarks, size=128)

            # Prepare inswapper input: blobFromImage with mean=0, std=255, swapRB
            aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
            face_blob = cv2.dnn.blobFromImage(
                aligned_bgr, 1.0 / 255.0, (128, 128),
                (0.0, 0.0, 0.0), swapRB=True,
            )
            emb_blob = latent.astype(np.float32)

            # Run inswapper
            input_names = list(self._inswapper.inputs.keys())
            outputs = self._inswapper.infer({
                input_names[0]: face_blob,
                input_names[1]: emb_blob,
            })
            swapped = list(outputs.values())[0][0].transpose(1, 2, 0)
            swapped = (swapped * 255).clip(0, 255).astype(np.uint8)

            # Paste back
            result = paste_back(swapped, result, M, size=128)

        return result
