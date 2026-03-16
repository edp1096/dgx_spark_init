"""TensorRT-based FaceSwap pipeline for ZIT.

Replaces InsightFace + onnxruntime with pure TensorRT inference.
Components:
  - SCRFD: Face detection (bounding boxes + 5-point landmarks)
  - ArcFace: Face embedding extraction (512-dim)
  - inswapper: Face replacement (128x128 aligned face + source embedding → swapped face)
  - CodeFormer: Face restoration (128px → 512px quality enhancement)

ONNX → TRT engine conversion is done on first use and cached as .engine files.
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

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


# ---------------------------------------------------------------------------
# CodeFormer face restoration
# ---------------------------------------------------------------------------
class CodeFormerRestorer:
    """Load and run CodeFormer for face restoration."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from codeformer import CodeFormer
        self._model = CodeFormer(
            dim_embd=512, n_head=8, n_layers=9,
            codebook_size=1024, latent_size=256,
            connect_list=['32', '64', '128', '256'],
            fix_modules=['quantize', 'generator'],
        )
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if "params_ema" in ckpt:
            self._model.load_state_dict(ckpt["params_ema"])
        elif "params" in ckpt:
            self._model.load_state_dict(ckpt["params"])
        else:
            self._model.load_state_dict(ckpt)
        self._model.eval().to(self.device)
        logger.info("CodeFormer loaded from %s", self.model_path)

    @torch.no_grad()
    def restore(self, face_rgb: np.ndarray, w: float = 0.7) -> np.ndarray:
        """Restore a face image (any size RGB) → 512x512 restored RGB.

        Args:
            face_rgb: Input face crop (RGB, uint8)
            w: Fidelity weight (0=quality, 1=identity preservation)
        """
        self._ensure_loaded()
        # Resize to 512x512 (CodeFormer's native resolution)
        inp = cv2.resize(face_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        # Normalize to [-1, 1], BCHW
        inp_t = torch.from_numpy(inp.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        inp_t = (inp_t - 0.5) / 0.5
        inp_t = inp_t.to(self.device)
        output, _, _ = self._model(inp_t, w=w, adain=True)
        # Back to uint8 RGB
        output = (output.squeeze(0).clamp(-1, 1) * 0.5 + 0.5) * 255.0
        output = output.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
        return output


# ---------------------------------------------------------------------------
# Face mask utilities (for inpaint refinement)
# ---------------------------------------------------------------------------
def create_face_mask(image_shape: tuple, bbox: np.ndarray, landmarks: np.ndarray,
                     padding: float = 1.5) -> np.ndarray:
    """Create a soft face mask for inpaint refinement.

    Args:
        image_shape: (H, W, C) of the target image
        bbox: [x1, y1, x2, y2] face bounding box
        landmarks: 5x2 landmarks
        padding: bbox expansion factor

    Returns:
        L-mode mask (uint8, 0=keep, 255=inpaint) suitable for ZIT inpaint pipeline
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # Expand bbox
    bw *= padding
    bh *= padding
    x1 = max(0, int(cx - bw / 2))
    y1 = max(0, int(cy - bh / 2))
    x2 = min(w, int(cx + bw / 2))
    y2 = min(h, int(cy + bh / 2))

    mask = np.zeros((h, w), dtype=np.uint8)
    # Draw filled ellipse in expanded bbox
    ecx, ecy = (x1 + x2) // 2, (y1 + y2) // 2
    eax, eay = (x2 - x1) // 2, (y2 - y1) // 2
    cv2.ellipse(mask, (ecx, ecy), (eax, eay), 0, 0, 360, 255, -1)
    # Feather edges
    ksize = max(3, int(min(eax, eay) * 0.3) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return mask


def create_detail_masks(landmarks: np.ndarray, image_shape: tuple) -> list[np.ndarray]:
    """Create individual masks for eyes, nose, mouth from 5-point landmarks.

    Returns list of (mask, label) tuples for FaceDetailer.
    """
    h, w = image_shape[:2]
    # landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    le, re, nose, lm, rm = landmarks

    eye_dist = np.linalg.norm(re - le)
    eye_r = int(eye_dist * 0.35)
    nose_r = int(eye_dist * 0.3)
    mouth_cx = int((lm[0] + rm[0]) / 2)
    mouth_cy = int((lm[1] + rm[1]) / 2)
    mouth_rx = int(np.linalg.norm(rm - lm) * 0.5)
    mouth_ry = int(mouth_rx * 0.6)

    parts = []
    # Left eye
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(le[0]), int(le[1])), eye_r, 255, -1)
    m = cv2.GaussianBlur(m, (0, 0), eye_r * 0.3)
    parts.append((m, "left_eye"))
    # Right eye
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(re[0]), int(re[1])), eye_r, 255, -1)
    m = cv2.GaussianBlur(m, (0, 0), eye_r * 0.3)
    parts.append((m, "right_eye"))
    # Nose
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(nose[0]), int(nose[1])), nose_r, 255, -1)
    m = cv2.GaussianBlur(m, (0, 0), nose_r * 0.3)
    parts.append((m, "nose"))
    # Mouth
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(m, (mouth_cx, mouth_cy), (mouth_rx, mouth_ry), 0, 0, 360, 255, -1)
    m = cv2.GaussianBlur(m, (0, 0), mouth_ry * 0.3)
    parts.append((m, "mouth"))

    return parts


def paste_back(
    swapped_face: np.ndarray, target: np.ndarray, M: np.ndarray,
    size: int = 128, blend_mode: str = "seamless", mask_blur: float = 0.3,
) -> np.ndarray:
    """Paste swapped face back onto target image.

    Args:
        blend_mode: "seamless" (Poisson) or "alpha" (feathered alpha blend)
        mask_blur: blur strength for mask edges (0.0 ~ 1.0, maps to GaussianBlur sigma)
    """
    M_inv = cv2.invertAffineTransform(M)

    # Create elliptical mask on aligned face space
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    axes = (int(size * 0.42), int(size * 0.48))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Apply blur to mask edges
    blur_sigma = max(1.0, mask_blur * size * 0.15)
    mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)

    # Warp swapped face and mask back to target space
    face_back = cv2.warpAffine(swapped_face, M_inv, (target.shape[1], target.shape[0]))
    mask_back = cv2.warpAffine(mask, M_inv, (target.shape[1], target.shape[0]))

    ys, xs = np.where(mask_back > 0)
    if len(xs) == 0 or len(ys) == 0:
        return target

    if blend_mode == "seamless":
        clone_center = (int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2))
        target_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        face_back_bgr = cv2.cvtColor(face_back, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.seamlessClone(face_back_bgr, target_bgr, mask_back, clone_center, cv2.NORMAL_CLONE)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Alpha blend
        mask_f = mask_back.astype(np.float32)[:, :, np.newaxis] / 255.0
        result = target.astype(np.float32) * (1 - mask_f) + face_back.astype(np.float32) * mask_f
        return result.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# FaceSwap Pipeline
# ---------------------------------------------------------------------------
class FaceSwapPipeline:
    """Complete face swap: detect → align → recognize → swap → paste back."""

    def __init__(self, model_dir: str):
        from zit_config import FACESWAP_DIR, SCRFD_FILE, ARCFACE_FILE, INSWAPPER_FILE, CODEFORMER_FILE

        self.model_dir = model_dir
        fs_dir = os.path.join(model_dir, FACESWAP_DIR)

        self.scrfd_path = os.path.join(fs_dir, SCRFD_FILE)
        self.arcface_path = os.path.join(fs_dir, ARCFACE_FILE)
        self.inswapper_path = os.path.join(fs_dir, INSWAPPER_FILE)
        self.codeformer_path = os.path.join(fs_dir, CODEFORMER_FILE)

        self._scrfd = None
        self._arcface = None
        self._inswapper = None
        self._emap = None
        self._codeformer = None

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

    def _ensure_codeformer_loaded(self):
        if self._codeformer is not None:
            return
        if not os.path.exists(self.codeformer_path):
            logger.warning("CodeFormer model not found at %s — face restoration disabled", self.codeformer_path)
            return
        self._codeformer = CodeFormerRestorer(self.codeformer_path)

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
        det_thresh: float = 0.5,
        blend_mode: str = "seamless",
        mask_blur: float = 0.3,
        face_index: int = 0,
        enable_restore: bool = True,
        codeformer_w: float = 0.7,
    ) -> tuple[np.ndarray, list]:
        """Swap face from source onto target.

        Args:
            target_image: RGB image with face to replace
            source_image: RGB image with source face
            det_thresh: SCRFD detection confidence threshold
            blend_mode: "seamless" (Poisson) or "alpha" (feathered)
            mask_blur: mask edge blur strength (0.0~1.0)
            face_index: which target face to swap (0=largest, -1=all)
            enable_restore: run CodeFormer on swapped face before paste
            codeformer_w: fidelity weight (0=quality, 1=identity)

        Returns:
            (result_image, swapped_face_info) — info contains (bbox, landmarks) for each swapped face
        """
        self._ensure_loaded()

        if enable_restore:
            self._ensure_codeformer_loaded()

        # Detect faces
        target_faces = self.detect_faces(target_image, threshold=det_thresh)
        source_faces = self.detect_faces(source_image, threshold=det_thresh)

        if not target_faces:
            raise ValueError("No face detected in target image")
        if not source_faces:
            raise ValueError("No face detected in source image")

        # Get source embedding and transform via emap
        src_bbox, src_landmarks = source_faces[0]
        source_embedding = self.get_embedding(source_image, src_landmarks)
        latent = np.dot(source_embedding.reshape(1, -1), self._emap)
        latent /= (np.linalg.norm(latent) + 1e-8)

        # Select target faces
        if face_index >= 0 and face_index < len(target_faces):
            swap_targets = [target_faces[face_index]]
        else:
            swap_targets = target_faces

        # Swap selected target faces
        result = target_image.copy()
        swapped_faces_info = []

        for bbox, landmarks in swap_targets:
            aligned, M = align_face(result, landmarks, size=128)

            # Prepare inswapper input
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

            # CodeFormer restoration: 128px → 512px → resize back to 128 for paste
            if enable_restore and self._codeformer is not None:
                logger.info("Running CodeFormer restoration (w=%.2f)...", codeformer_w)
                restored_512 = self._codeformer.restore(swapped, w=codeformer_w)
                swapped = cv2.resize(restored_512, (128, 128), interpolation=cv2.INTER_LANCZOS4)

            # Paste back
            result = paste_back(swapped, result, M, size=128, blend_mode=blend_mode, mask_blur=mask_blur)
            swapped_faces_info.append((bbox, landmarks))

        return result, swapped_faces_info
