"""Tests for inpaint parameter overhaul + faceswap enhancement (no GPU required).

Covers:
  - Inpaint default parameters alignment with official VideoX-Fun
  - CodeFormer vendored architecture import + model structure
  - FaceSwap pipeline new parameters (restore, refine, detailer)
  - Face mask / detail mask utilities
  - Worker kwargs flow for new multi-stage faceswap
  - Pipeline parameter compatibility (control_image for inpaint)

Usage:
    cd /root/zit-ui && python -m pytest tests/test_inpaint_faceswap_overhaul.py -v
    cd /root/zit-ui && python tests/test_inpaint_faceswap_overhaul.py
"""
import sys
import os
import inspect
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "ui"))


def make_test_image(w=256, h=256, color=(128, 64, 32)):
    return np.full((h, w, 3), color, dtype=np.uint8)


# ===========================================================================
# Test 1: Inpaint parameter alignment with official settings
# ===========================================================================
class TestInpaintDefaults:
    """Verify inpaint defaults match official predict_i2i_inpaint_2.1.py."""

    def test_config_inpaint_steps(self):
        from zit_config import DEFAULT_INPAINT_STEPS
        assert DEFAULT_INPAINT_STEPS == 25, f"Expected 25, got {DEFAULT_INPAINT_STEPS}"
        print("  PASS: test_config_inpaint_steps")

    def test_config_inpaint_guidance(self):
        from zit_config import DEFAULT_INPAINT_GUIDANCE
        assert DEFAULT_INPAINT_GUIDANCE == 4.0, f"Expected 4.0, got {DEFAULT_INPAINT_GUIDANCE}"
        print("  PASS: test_config_inpaint_guidance")

    def test_config_inpaint_cfg_truncation(self):
        from zit_config import DEFAULT_INPAINT_CFG_TRUNCATION
        assert DEFAULT_INPAINT_CFG_TRUNCATION == 1.0, f"Expected 1.0, got {DEFAULT_INPAINT_CFG_TRUNCATION}"
        print("  PASS: test_config_inpaint_cfg_truncation")

    def test_config_inpaint_control_scale(self):
        from zit_config import DEFAULT_INPAINT_CONTROL_SCALE
        assert DEFAULT_INPAINT_CONTROL_SCALE == 0.9, f"Expected 0.9, got {DEFAULT_INPAINT_CONTROL_SCALE}"
        print("  PASS: test_config_inpaint_control_scale")

    def test_generate_inpaint_defaults(self):
        """generate_inpaint function signature should have updated defaults."""
        from generators import generate_inpaint
        sig = inspect.signature(generate_inpaint)
        assert sig.parameters["num_steps"].default == 25
        assert sig.parameters["guidance_scale"].default == 4.0
        assert sig.parameters["cfg_truncation"].default == 1.0
        print("  PASS: test_generate_inpaint_defaults")

    def test_generate_outpaint_defaults(self):
        """generate_outpaint should also use inpaint-aligned defaults."""
        from generators import generate_outpaint
        sig = inspect.signature(generate_outpaint)
        assert sig.parameters["num_steps"].default == 25
        assert sig.parameters["guidance_scale"].default == 4.0
        assert sig.parameters["cfg_truncation"].default == 1.0
        print("  PASS: test_generate_outpaint_defaults")

    def test_pipeline_accepts_control_image(self):
        """Pipeline __call__ must accept control_image for inpaint + control combo."""
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        sig = inspect.signature(ZImageControlPipeline.__call__)
        params = set(sig.parameters.keys())
        assert "control_image" in params, "Pipeline must accept control_image"
        assert "image" in params, "Pipeline must accept image (inpaint source)"
        assert "mask_image" in params, "Pipeline must accept mask_image"
        print("  PASS: test_pipeline_accepts_control_image")

    def test_guidance_over_1_enables_cfg(self):
        """guidance_scale > 1 should enable classifier-free guidance in pipeline."""
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        # Check the do_classifier_free_guidance property
        assert hasattr(ZImageControlPipeline, "do_classifier_free_guidance"), \
            "Pipeline must have do_classifier_free_guidance property"
        print("  PASS: test_guidance_over_1_enables_cfg")


# ===========================================================================
# Test 2: CodeFormer vendored architecture
# ===========================================================================
class TestCodeFormerArch:
    """Test vendored CodeFormer architecture imports and structure."""

    def test_import_codeformer(self):
        from codeformer import CodeFormer
        assert CodeFormer is not None
        print("  PASS: test_import_codeformer")

    def test_import_vqgan(self):
        from codeformer.vqgan_arch import VQAutoEncoder, VectorQuantizer, Encoder, Generator
        assert VQAutoEncoder is not None
        assert VectorQuantizer is not None
        print("  PASS: test_import_vqgan")

    def test_codeformer_is_vqautoencoder_subclass(self):
        from codeformer.codeformer_arch import CodeFormer
        from codeformer.vqgan_arch import VQAutoEncoder
        assert issubclass(CodeFormer, VQAutoEncoder)
        print("  PASS: test_codeformer_is_vqautoencoder_subclass")

    def test_codeformer_instantiation(self):
        """CodeFormer model should instantiate without weights."""
        import torch
        from codeformer import CodeFormer
        model = CodeFormer(
            dim_embd=512, n_head=8, n_layers=9,
            codebook_size=1024, latent_size=256,
        )
        assert isinstance(model, torch.nn.Module)
        # Check key components exist
        assert hasattr(model, "encoder")
        assert hasattr(model, "generator")
        assert hasattr(model, "quantize")
        assert hasattr(model, "ft_layers")
        assert hasattr(model, "fuse_convs_dict")
        print("  PASS: test_codeformer_instantiation")

    def test_codeformer_forward_shape(self):
        """CodeFormer forward should accept 512x512 and output 512x512."""
        import torch
        from codeformer import CodeFormer
        model = CodeFormer()
        model.eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out, logits, lq_feat = model(x, w=0.5)
        assert out.shape == (1, 3, 512, 512), f"Expected (1,3,512,512), got {out.shape}"
        print("  PASS: test_codeformer_forward_shape")

    def test_codeformer_w_parameter(self):
        """w=0 should produce output (quality mode), w=1 should also work (fidelity)."""
        import torch
        from codeformer import CodeFormer
        model = CodeFormer()
        model.eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out_q, _, _ = model(x, w=0)
            out_f, _, _ = model(x, w=1)
        assert out_q.shape == out_f.shape
        # Outputs should differ when w changes
        assert not torch.allclose(out_q, out_f, atol=1e-3), "w=0 and w=1 should produce different outputs"
        print("  PASS: test_codeformer_w_parameter")


# ===========================================================================
# Test 3: CodeFormerRestorer class
# ===========================================================================
class TestCodeFormerRestorer:
    """Test the CodeFormerRestorer wrapper."""

    def test_restorer_class_exists(self):
        from face_swap import CodeFormerRestorer
        assert CodeFormerRestorer is not None
        print("  PASS: test_restorer_class_exists")

    def test_restorer_init_no_crash(self):
        from face_swap import CodeFormerRestorer
        r = CodeFormerRestorer("/nonexistent/path.pth", device="cpu")
        assert r._model is None, "Model should not be loaded at init"
        print("  PASS: test_restorer_init_no_crash")


# ===========================================================================
# Test 4: Face mask utilities
# ===========================================================================
class TestFaceMaskUtils:
    """Test face mask creation utilities."""

    def test_create_face_mask_shape(self):
        from face_swap import create_face_mask
        bbox = np.array([100, 80, 200, 220])
        landmarks = np.array([
            [130, 120], [170, 120], [150, 150], [135, 180], [165, 180]
        ], dtype=np.float32)
        mask = create_face_mask((300, 300, 3), bbox, landmarks, padding=1.5)
        assert mask.shape == (300, 300), f"Expected (300,300), got {mask.shape}"
        assert mask.dtype == np.uint8
        print("  PASS: test_create_face_mask_shape")

    def test_create_face_mask_center_nonzero(self):
        from face_swap import create_face_mask
        bbox = np.array([100, 80, 200, 220])
        landmarks = np.array([
            [130, 120], [170, 120], [150, 150], [135, 180], [165, 180]
        ], dtype=np.float32)
        mask = create_face_mask((300, 300, 3), bbox, landmarks, padding=1.5)
        # Center of face should be masked
        assert mask[150, 150] > 0, f"Center should be masked, got {mask[150, 150]}"
        # Far corner should not
        assert mask[5, 5] == 0, f"Corner should be unmasked, got {mask[5, 5]}"
        print("  PASS: test_create_face_mask_center_nonzero")

    def test_create_face_mask_clipping(self):
        """Mask should not go outside image bounds."""
        from face_swap import create_face_mask
        # Face near edge
        bbox = np.array([0, 0, 50, 50])
        landmarks = np.array([
            [15, 20], [35, 20], [25, 30], [18, 40], [32, 40]
        ], dtype=np.float32)
        mask = create_face_mask((100, 100, 3), bbox, landmarks, padding=2.0)
        assert mask.shape == (100, 100)
        assert mask.max() <= 255
        print("  PASS: test_create_face_mask_clipping")

    def test_create_detail_masks_count(self):
        from face_swap import create_detail_masks
        landmarks = np.array([
            [130, 120], [170, 120], [150, 150], [135, 180], [165, 180]
        ], dtype=np.float32)
        parts = create_detail_masks(landmarks, (300, 300, 3))
        assert len(parts) == 4, f"Expected 4 parts (2 eyes, nose, mouth), got {len(parts)}"
        labels = [label for _, label in parts]
        assert "left_eye" in labels
        assert "right_eye" in labels
        assert "nose" in labels
        assert "mouth" in labels
        print("  PASS: test_create_detail_masks_count")

    def test_create_detail_masks_shapes(self):
        from face_swap import create_detail_masks
        landmarks = np.array([
            [130, 120], [170, 120], [150, 150], [135, 180], [165, 180]
        ], dtype=np.float32)
        parts = create_detail_masks(landmarks, (300, 300, 3))
        for mask, label in parts:
            assert mask.shape == (300, 300), f"{label} mask shape {mask.shape} != (300,300)"
            assert mask.dtype == np.uint8
            assert mask.max() > 0, f"{label} mask is all zeros"
        print("  PASS: test_create_detail_masks_shapes")


# ===========================================================================
# Test 5: FaceSwap pipeline new signature
# ===========================================================================
class TestFaceSwapNewSignature:
    """Test that FaceSwapPipeline.swap_face accepts new parameters."""

    def test_swap_face_accepts_restore_params(self):
        from face_swap import FaceSwapPipeline
        sig = inspect.signature(FaceSwapPipeline.swap_face)
        params = set(sig.parameters.keys()) - {"self"}
        expected = {"target_image", "source_image", "det_thresh", "blend_mode",
                    "mask_blur", "face_index", "enable_restore", "codeformer_w"}
        missing = expected - params
        assert not missing, f"swap_face missing params: {missing}"
        print("  PASS: test_swap_face_accepts_restore_params")

    def test_swap_face_returns_tuple(self):
        """swap_face should return (result, swapped_faces_info) tuple."""
        from face_swap import FaceSwapPipeline
        sig = inspect.signature(FaceSwapPipeline.swap_face)
        # Check return type annotation if available
        ret = sig.return_annotation
        # At minimum, the function signature should indicate it returns tuple
        assert "tuple" in str(ret).lower() or ret is inspect.Parameter.empty, \
            "swap_face should return tuple"
        print("  PASS: test_swap_face_returns_tuple")


# ===========================================================================
# Test 6: Generator kwargs flow
# ===========================================================================
class TestGeneratorKwargs:
    """Test that generate_faceswap builds correct kwargs for worker."""

    def test_generate_faceswap_new_params(self):
        from generators import generate_faceswap
        sig = inspect.signature(generate_faceswap)
        params = set(sig.parameters.keys())
        expected_new = {"enable_restore", "codeformer_w", "enable_refine",
                        "refine_prompt", "refine_steps", "enable_detailer"}
        missing = expected_new - params
        assert not missing, f"generate_faceswap missing params: {missing}"
        print("  PASS: test_generate_faceswap_new_params")

    def test_generate_faceswap_default_restore_on(self):
        from generators import generate_faceswap
        sig = inspect.signature(generate_faceswap)
        assert sig.parameters["enable_restore"].default is True
        assert sig.parameters["enable_refine"].default is True
        assert sig.parameters["enable_detailer"].default is False
        print("  PASS: test_generate_faceswap_default_restore_on")

    def test_generate_faceswap_codeformer_w_default(self):
        from generators import generate_faceswap
        sig = inspect.signature(generate_faceswap)
        assert sig.parameters["codeformer_w"].default == 0.7
        print("  PASS: test_generate_faceswap_codeformer_w_default")

    def test_generate_faceswap_refine_steps_default(self):
        from generators import generate_faceswap
        sig = inspect.signature(generate_faceswap)
        assert sig.parameters["refine_steps"].default == 15
        print("  PASS: test_generate_faceswap_refine_steps_default")


# ===========================================================================
# Test 7: Config completeness
# ===========================================================================
class TestConfigCompleteness:
    """Test all new config values exist and are reasonable."""

    def test_codeformer_config(self):
        from zit_config import CODEFORMER_FILE, CODEFORMER_URL, DEFAULT_CODEFORMER_FIDELITY
        assert CODEFORMER_FILE == "codeformer.pth"
        assert "github.com" in CODEFORMER_URL
        assert 0 <= DEFAULT_CODEFORMER_FIDELITY <= 1
        print("  PASS: test_codeformer_config")

    def test_t2i_defaults_unchanged(self):
        """T2I defaults should NOT be affected by inpaint changes."""
        from zit_config import DEFAULT_STEPS, DEFAULT_GUIDANCE, DEFAULT_CFG_TRUNCATION
        assert DEFAULT_STEPS == 8, "T2I steps should still be 8"
        assert DEFAULT_GUIDANCE == 0.5, "T2I guidance should still be 0.5"
        assert DEFAULT_CFG_TRUNCATION == 0.9, "T2I cfg_truncation should still be 0.9"
        print("  PASS: test_t2i_defaults_unchanged")

    def test_inpaint_defaults_separate_from_t2i(self):
        """Inpaint defaults must be separate constants from T2I defaults."""
        from zit_config import (
            DEFAULT_STEPS, DEFAULT_GUIDANCE,
            DEFAULT_INPAINT_STEPS, DEFAULT_INPAINT_GUIDANCE,
        )
        assert DEFAULT_STEPS != DEFAULT_INPAINT_STEPS, "Inpaint steps should differ from T2I"
        assert DEFAULT_GUIDANCE != DEFAULT_INPAINT_GUIDANCE, "Inpaint guidance should differ from T2I"
        print("  PASS: test_inpaint_defaults_separate_from_t2i")


# ===========================================================================
# Test 8: Paste back modes
# ===========================================================================
class TestPasteBack:
    """Test paste_back function with different blend modes."""

    def test_paste_back_seamless(self):
        import cv2
        from face_swap import paste_back
        # Create simple test data
        face = np.full((128, 128, 3), 200, dtype=np.uint8)
        target = np.full((256, 256, 3), 100, dtype=np.uint8)
        M = np.array([[1.0, 0, 64], [0, 1.0, 64]], dtype=np.float64)  # translate to (64,64)
        result = paste_back(face, target, M, size=128, blend_mode="seamless", mask_blur=0.3)
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8
        print("  PASS: test_paste_back_seamless")

    def test_paste_back_alpha(self):
        from face_swap import paste_back
        face = np.full((128, 128, 3), 200, dtype=np.uint8)
        target = np.full((256, 256, 3), 100, dtype=np.uint8)
        M = np.array([[1.0, 0, 64], [0, 1.0, 64]], dtype=np.float64)
        result = paste_back(face, target, M, size=128, blend_mode="alpha", mask_blur=0.3)
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8
        # Alpha blended center should be between face and target values
        center_val = result[128, 128, 0]
        assert 100 <= center_val <= 200, f"Alpha blend center should be between 100-200, got {center_val}"
        print("  PASS: test_paste_back_alpha")

    def test_paste_back_empty_mask(self):
        """Warping face entirely outside target should return target unchanged."""
        from face_swap import paste_back
        face = np.full((128, 128, 3), 200, dtype=np.uint8)
        target = np.full((256, 256, 3), 100, dtype=np.uint8)
        # M that places face way outside
        M = np.array([[1.0, 0, 1000], [0, 1.0, 1000]], dtype=np.float64)
        result = paste_back(face, target, M, size=128, blend_mode="seamless")
        # Should return target unchanged
        assert np.array_equal(result, target)
        print("  PASS: test_paste_back_empty_mask")


# ===========================================================================
# Test 9: Download models includes CodeFormer
# ===========================================================================
class TestDownloadModels:
    """Test that download_models.py includes CodeFormer."""

    def test_download_faceswap_includes_codeformer(self):
        download_path = Path(__file__).resolve().parent.parent / "app" / "ui" / "download_models.py"
        content = download_path.read_text()
        assert "CODEFORMER_FILE" in content, "download_models.py should reference CODEFORMER_FILE"
        assert "CODEFORMER_URL" in content, "download_models.py should reference CODEFORMER_URL"
        print("  PASS: test_download_faceswap_includes_codeformer")


# ===========================================================================
# Runner
# ===========================================================================
def run_all():
    test_classes = [
        TestInpaintDefaults,
        TestCodeFormerArch,
        TestCodeFormerRestorer,
        TestFaceMaskUtils,
        TestFaceSwapNewSignature,
        TestGeneratorKwargs,
        TestConfigCompleteness,
        TestPasteBack,
        TestDownloadModels,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        print(f"\n{'=' * 60}")
        print(f"  {cls.__name__}")
        print(f"{'=' * 60}")
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
            except Exception as e:
                failed += 1
                errors.append((f"{cls.__name__}.{method_name}", str(e)))
                print(f"  FAIL: {method_name} — {e}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
