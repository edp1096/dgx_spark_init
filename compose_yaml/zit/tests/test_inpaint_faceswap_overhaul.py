"""Tests for inpaint parameter overhaul + faceswap Phase 5 (no GPU required).

Covers:
  - Inpaint default parameters alignment with official VideoX-Fun
  - CodeFormer vendored architecture import + model structure
  - FaceSwap Phase 5: SCRFD + auto-mask + inpaint delegation
  - Generator kwargs flow for new faceswap
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
        assert not torch.allclose(out_q, out_f, atol=1e-3), "w=0 and w=1 should produce different outputs"
        print("  PASS: test_codeformer_w_parameter")


# ===========================================================================
# Test 3: FaceSwap Phase 5 — SCRFD + auto-mask
# ===========================================================================
class TestFaceSwapPhase5:
    """Test the new SCRFD-based face detection and auto-mask approach."""

    def test_scrfd_detector_class_exists(self):
        from face_swap import SCRFDDetector
        assert SCRFDDetector is not None
        print("  PASS: test_scrfd_detector_class_exists")

    def test_create_face_mask_function_exists(self):
        from face_swap import create_face_mask
        sig = inspect.signature(create_face_mask)
        params = set(sig.parameters.keys())
        assert "image" in params
        assert "model_dir" in params
        assert "face_index" in params
        assert "padding" in params
        assert "det_threshold" in params
        print("  PASS: test_create_face_mask_function_exists")

    def test_preview_face_detection_exists(self):
        from face_swap import preview_face_detection
        sig = inspect.signature(preview_face_detection)
        params = set(sig.parameters.keys())
        assert "image" in params
        assert "model_dir" in params
        print("  PASS: test_preview_face_detection_exists")

    def test_get_detector_singleton_exists(self):
        from face_swap import get_detector
        assert callable(get_detector)
        print("  PASS: test_get_detector_singleton_exists")

    def test_no_old_trt_classes(self):
        """Old TRT classes should be removed from face_swap."""
        import face_swap
        assert not hasattr(face_swap, "FaceSwapPipeline"), "FaceSwapPipeline should be removed"
        assert not hasattr(face_swap, "CodeFormerRestorer"), "CodeFormerRestorer should be removed"
        assert not hasattr(face_swap, "TRTEngine"), "TRTEngine should be removed"
        assert not hasattr(face_swap, "paste_back"), "paste_back should be removed"
        assert not hasattr(face_swap, "create_detail_masks"), "create_detail_masks should be removed"
        print("  PASS: test_no_old_trt_classes")


# ===========================================================================
# Test 4: Generator kwargs flow
# ===========================================================================
class TestGeneratorKwargs:
    """Test that generate_faceswap builds correct kwargs for new pipeline."""

    def test_generate_faceswap_new_params(self):
        from generators import generate_faceswap
        sig = inspect.signature(generate_faceswap)
        params = set(sig.parameters.keys())
        expected_new = {"image", "prompt", "face_index", "padding", "det_threshold",
                        "num_steps", "guidance_scale", "control_scale", "seed"}
        missing = expected_new - params
        assert not missing, f"generate_faceswap missing params: {missing}"
        print("  PASS: test_generate_faceswap_new_params")

    def test_generate_faceswap_no_old_params(self):
        from generators import generate_faceswap
        sig = inspect.signature(generate_faceswap)
        params = set(sig.parameters.keys())
        old_params = {"target_image", "source_image", "enable_restore",
                      "codeformer_w", "enable_refine", "enable_detailer",
                      "blend_mode", "mask_blur"}
        unexpected = old_params & params
        assert not unexpected, f"Old TRT params still present: {unexpected}"
        print("  PASS: test_generate_faceswap_no_old_params")

    def test_generate_faceswap_defaults(self):
        from generators import generate_faceswap
        sig = inspect.signature(generate_faceswap)
        assert sig.parameters["face_index"].default == 0
        assert sig.parameters["padding"].default == 1.3
        assert sig.parameters["det_threshold"].default == 0.5
        assert sig.parameters["num_steps"].default == 25
        assert sig.parameters["guidance_scale"].default == 4.0
        print("  PASS: test_generate_faceswap_defaults")

    def test_generate_faceswap_raises_on_none_image(self):
        from generators import generate_faceswap
        try:
            generate_faceswap(None, "a face")
            assert False, "Should have raised"
        except Exception as e:
            assert "image" in str(e).lower() or "required" in str(e).lower()
        print("  PASS: test_generate_faceswap_raises_on_none_image")


# ===========================================================================
# Test 5: Config completeness
# ===========================================================================
class TestConfigCompleteness:
    """Test all config values exist and are reasonable."""

    def test_scrfd_config(self):
        from zit_config import SCRFD_FILE, SCRFD_URL
        assert SCRFD_FILE == "scrfd_10g_bnkps.onnx"
        assert "huggingface.co" in SCRFD_URL
        print("  PASS: test_scrfd_config")

    def test_t2i_defaults_unchanged(self):
        """T2I defaults should NOT be affected by faceswap changes."""
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

    def test_required_models_faceswap(self):
        from pipeline_manager import REQUIRED_MODELS
        required = REQUIRED_MODELS.get("faceswap", [])
        assert any("scrfd" in r for r in required), f"Should require SCRFD: {required}"
        assert not any("faceswap/" in r for r in required), "Should not require old faceswap/ dir"
        print("  PASS: test_required_models_faceswap")


# ===========================================================================
# Test 6: Download models includes SCRFD
# ===========================================================================
class TestDownloadModels:
    """Test that download_models.py includes SCRFD."""

    def test_download_includes_scrfd(self):
        download_path = Path(__file__).resolve().parent.parent / "app" / "ui" / "download_models.py"
        content = download_path.read_text()
        assert "SCRFD_FILE" in content or "scrfd" in content.lower(), \
            "download_models.py should reference SCRFD"
        print("  PASS: test_download_includes_scrfd")

    def test_download_no_old_faceswap(self):
        download_path = Path(__file__).resolve().parent.parent / "app" / "ui" / "download_models.py"
        content = download_path.read_text()
        assert "ARCFACE" not in content, "download_models.py should not reference ARCFACE"
        assert "INSWAPPER" not in content, "download_models.py should not reference INSWAPPER"
        print("  PASS: test_download_no_old_faceswap")


# ===========================================================================
# Runner
# ===========================================================================
def run_all():
    test_classes = [
        TestInpaintDefaults,
        TestCodeFormerArch,
        TestFaceSwapPhase5,
        TestGeneratorKwargs,
        TestConfigCompleteness,
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
