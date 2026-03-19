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
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError(f"Failed to build TRT engine for {onnx_path}")

        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        logger.info("TRT engine saved: %s", engine_path)

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with numpy arrays."""
        import torch

        # Allocate output buffers
        output_buffers = {}
        for name, info in self.outputs.items():
            shape = info["shape"]
            # Handle dynamic dims (-1) by using context binding shape
            output_buffers[name] = torch.empty(
                shape, dtype=torch.float32, device="cuda"
            )

        # Set input tensors
        for name, arr in inputs.items():
            t = torch.from_numpy(arr).cuda()
            self.context.set_input_shape(name, t.shape)
            self.context.set_tensor_address(name, t.data_ptr())

        # Set output addresses
        for name, buf in output_buffers.items():
            self.context.set_tensor_address(name, buf.data_ptr())

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

    def _ensure_loaded(self):
        if self._scrfd is not None:
            return

        for path in [self.scrfd_path, self.arcface_path, self.inswapper_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"FaceSwap model not found: {path}")

        logger.info("Loading FaceSwap TRT engines...")
        self._scrfd = TRTEngine(self.scrfd_path)
        self._arcface = TRTEngine(self.arcface_path)
        self._inswapper = TRTEngine(self.inswapper_path)
        logger.info("FaceSwap engines loaded")

    def detect_faces(self, image: np.ndarray, threshold: float = 0.5):
        """Detect faces using SCRFD. Returns list of (bbox, landmarks_5)."""
        self._ensure_loaded()

        # Preprocess: resize to 640x640, normalize
        h, w = image.shape[:2]
        scale = min(640 / w, 640 / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh))
        padded = np.zeros((640, 640, 3), dtype=np.float32)
        padded[:nh, :nw] = resized.astype(np.float32)
        blob = padded.transpose(2, 0, 1)[np.newaxis] / 255.0  # NCHW, [0-1]

        # Run SCRFD
        input_name = list(self._scrfd.inputs.keys())[0]
        outputs = self._scrfd.infer({input_name: blob.astype(np.float32)})

        # TODO: Parse SCRFD outputs (bboxes, scores, landmarks)
        # This is model-specific and depends on the exact SCRFD variant.
        # For now, return placeholder — needs testing on DGX.
        logger.warning("SCRFD output parsing not yet implemented — needs DGX testing")
        return []

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

        # Get source embedding
        src_bbox, src_landmarks = source_faces[0]
        source_embedding = self.get_embedding(source_image, src_landmarks)

        # Swap each target face
        result = target_image.copy()
        for bbox, landmarks in target_faces:
            aligned, M = align_face(result, landmarks, size=128)

            # Prepare inswapper input
            face_blob = aligned.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
            emb_blob = source_embedding[np.newaxis].astype(np.float32)

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
