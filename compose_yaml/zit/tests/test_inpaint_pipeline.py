"""Inpaint pipeline parameter tests (no GPU required).

Tests:
  1. Inpaint default parameters match official settings
  2. Pipeline __call__ signature accepts required params
  3. Config completeness

Usage:
    cd /root/zit-ui && python tests/test_inpaint_pipeline.py
    cd /root/zit-ui && python -m pytest tests/test_inpaint_pipeline.py -v
"""
import sys
import inspect
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "ui"))


# ===========================================================================
# Test 1: Inpaint default parameters
# ===========================================================================
class TestInpaintDefaults:
    """Verify inpaint defaults match official settings."""

    def test_config_inpaint_steps(self):
        from zit_config import DEFAULT_INPAINT_STEPS
        assert DEFAULT_INPAINT_STEPS == 25, f"Expected 25, got {DEFAULT_INPAINT_STEPS}"
        print("  PASS: inpaint steps = 25")

    def test_config_inpaint_guidance(self):
        from zit_config import DEFAULT_INPAINT_GUIDANCE
        assert DEFAULT_INPAINT_GUIDANCE == 4.0, f"Expected 4.0, got {DEFAULT_INPAINT_GUIDANCE}"
        print("  PASS: inpaint guidance = 4.0")

    def test_config_inpaint_cfg_truncation(self):
        from zit_config import DEFAULT_INPAINT_CFG_TRUNCATION
        assert DEFAULT_INPAINT_CFG_TRUNCATION == 1.0
        print("  PASS: inpaint cfg_truncation = 1.0")

    def test_config_inpaint_control_scale(self):
        from zit_config import DEFAULT_INPAINT_CONTROL_SCALE
        assert DEFAULT_INPAINT_CONTROL_SCALE == 0.9
        print("  PASS: inpaint control_scale = 0.9")

    def test_generate_inpaint_defaults(self):
        from generators import generate_inpaint
        sig = inspect.signature(generate_inpaint)
        assert sig.parameters["num_steps"].default == 25
        assert sig.parameters["guidance_scale"].default == 4.0
        assert sig.parameters["cfg_truncation"].default == 1.0
        print("  PASS: generate_inpaint signature defaults")

    def test_generate_outpaint_defaults(self):
        from generators import generate_outpaint
        sig = inspect.signature(generate_outpaint)
        assert sig.parameters["num_steps"].default == 25
        assert sig.parameters["guidance_scale"].default == 4.0
        assert sig.parameters["cfg_truncation"].default == 1.0
        print("  PASS: generate_outpaint signature defaults")


# ===========================================================================
# Test 2: Pipeline signature
# ===========================================================================
class TestPipelineSignature:
    """Verify ZImageControlPipeline accepts required params."""

    def test_pipeline_accepts_inpaint_params(self):
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        sig = inspect.signature(ZImageControlPipeline.__call__)
        params = set(sig.parameters.keys())
        assert "control_image" in params, "Pipeline must accept control_image"
        assert "image" in params, "Pipeline must accept image"
        assert "mask_image" in params, "Pipeline must accept mask_image"
        print("  PASS: pipeline accepts inpaint params")

    def test_pipeline_has_cfg_property(self):
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        assert hasattr(ZImageControlPipeline, "do_classifier_free_guidance")
        print("  PASS: pipeline has do_classifier_free_guidance")

    def test_pipeline_height_width_optional(self):
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        sig = inspect.signature(ZImageControlPipeline.__call__)
        h_param = sig.parameters.get("height")
        w_param = sig.parameters.get("width")
        assert h_param is not None and h_param.default is None
        assert w_param is not None and w_param.default is None
        print("  PASS: height/width optional (default None)")

    def test_worker_inpaint_params_compatible(self):
        """Params that _run_inpaint passes should exist in pipeline signature."""
        inpaint_params = {
            "prompt", "negative_prompt", "height", "width",
            "image", "mask_image", "control_context_scale",
            "num_inference_steps", "guidance_scale", "cfg_truncation",
            "max_sequence_length", "generator",
        }
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        sig = inspect.signature(ZImageControlPipeline.__call__)
        pipeline_params = set(sig.parameters.keys()) - {"self"}

        missing = inpaint_params - pipeline_params
        assert not missing, f"Worker passes params not in pipeline: {missing}"
        print("  PASS: worker inpaint params compatible")

    def test_cfg_normalization_has_default(self):
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        sig = inspect.signature(ZImageControlPipeline.__call__)
        cfg_norm = sig.parameters.get("cfg_normalization")
        assert cfg_norm is not None, "Pipeline should have cfg_normalization"
        assert cfg_norm.default is not inspect.Parameter.empty
        print("  PASS: cfg_normalization has default")


# ===========================================================================
# Test 3: Config completeness
# ===========================================================================
class TestConfigCompleteness:

    def test_t2i_defaults_unchanged(self):
        from zit_config import DEFAULT_STEPS, DEFAULT_GUIDANCE, DEFAULT_CFG_TRUNCATION
        assert DEFAULT_STEPS == 8
        assert DEFAULT_GUIDANCE == 0.5
        assert DEFAULT_CFG_TRUNCATION == 0.9
        print("  PASS: T2I defaults unchanged")

    def test_inpaint_defaults_separate_from_t2i(self):
        from zit_config import (DEFAULT_STEPS, DEFAULT_GUIDANCE,
                                DEFAULT_INPAINT_STEPS, DEFAULT_INPAINT_GUIDANCE)
        assert DEFAULT_STEPS != DEFAULT_INPAINT_STEPS
        assert DEFAULT_GUIDANCE != DEFAULT_INPAINT_GUIDANCE
        print("  PASS: inpaint defaults separate from T2I")

    def test_controlnet_config(self):
        from zit_config import CONTROLNET_CONFIG
        assert CONTROLNET_CONFIG["control_in_dim"] == 33
        print("  PASS: control_in_dim = 33")


# ===========================================================================
# Runner
# ===========================================================================
def run_all():
    test_classes = [
        TestInpaintDefaults,
        TestPipelineSignature,
        TestConfigCompleteness,
    ]

    total = passed = failed = 0
    errors = []

    for cls in test_classes:
        print(f"\n{'=' * 60}")
        print(f"  {cls.__name__}")
        print(f"{'=' * 60}")
        instance = cls()
        for method_name in sorted(m for m in dir(instance) if m.startswith("test_")):
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
            except Exception as e:
                failed += 1
                errors.append((f"{cls.__name__}.{method_name}", str(e)))
                print(f"  FAIL: {method_name} -- {e}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    if errors:
        for name, err in errors:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
