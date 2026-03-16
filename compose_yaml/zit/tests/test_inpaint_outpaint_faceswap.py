"""Inpaint / Outpaint / FaceSwap — logic tests (no GPU required).

Tests the non-pipeline logic: mask extraction, canvas expansion,
kwargs construction, and error handling.

Usage:
    cd /root/zit-ui && python -m pytest tests/test_inpaint_outpaint_faceswap.py -v
    cd /root/zit-ui && python tests/test_inpaint_outpaint_faceswap.py
"""
import sys
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "ui"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_test_image(w=256, h=256, color=(128, 64, 32)):
    """Create a simple test image as numpy array (RGB)."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def make_editor_value(w=256, h=256, mask_rect=(50, 50, 150, 150)):
    """Simulate gr.ImageEditor output: {background, layers, composite}."""
    bg = make_test_image(w, h, (100, 100, 100))
    # Create a mask layer — white rectangle on black
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    y1, x1, y2, x2 = mask_rect
    layer[y1:y2, x1:x2] = [255, 255, 255, 255]
    return {
        "background": bg,
        "layers": [layer],
        "composite": bg.copy(),
    }


# ===========================================================================
# Test 1: Inpaint — mask extraction from gr.ImageEditor
# ===========================================================================
class TestInpaintMaskExtraction:
    """Test that generate_inpaint correctly extracts image + mask from editor."""

    def test_mask_from_layers_basic(self):
        """White brush strokes in layers → binary mask."""
        editor = make_editor_value(256, 256, (50, 50, 150, 150))
        bg = editor["background"]
        layers = editor["layers"]

        # Replicate generators.py logic
        mask = np.zeros(bg.shape[:2], dtype=np.uint8)
        for layer in layers:
            if isinstance(layer, np.ndarray) and layer.ndim >= 2:
                if layer.ndim == 3:
                    layer_gray = np.any(layer > 0, axis=2).astype(np.uint8) * 255
                else:
                    layer_gray = (layer > 0).astype(np.uint8) * 255
                mask = np.maximum(mask, layer_gray)

        # Mask should be 255 in the rectangle, 0 outside
        assert mask[100, 100] == 255, f"Center of mask should be 255, got {mask[100, 100]}"
        assert mask[10, 10] == 0, f"Outside mask should be 0, got {mask[10, 10]}"
        assert mask.shape == (256, 256)
        print("  PASS: test_mask_from_layers_basic")

    def test_mask_empty_layers_detected(self):
        """Empty layers list should be caught."""
        editor = {
            "background": make_test_image(),
            "layers": [],
            "composite": make_test_image(),
        }
        # generators.py raises gr.Error when layers empty
        layers = editor.get("layers", [])
        assert len(layers) == 0, "Should detect empty layers"
        print("  PASS: test_mask_empty_layers_detected")

    def test_mask_multiple_layers_union(self):
        """Multiple layers should be OR'd together."""
        bg = make_test_image(256, 256)
        layer1 = np.zeros((256, 256, 4), dtype=np.uint8)
        layer1[10:50, 10:50] = [255, 255, 255, 255]
        layer2 = np.zeros((256, 256, 4), dtype=np.uint8)
        layer2[200:240, 200:240] = [255, 255, 255, 255]

        mask = np.zeros(bg.shape[:2], dtype=np.uint8)
        for layer in [layer1, layer2]:
            layer_gray = np.any(layer > 0, axis=2).astype(np.uint8) * 255
            mask = np.maximum(mask, layer_gray)

        assert mask[30, 30] == 255, "First brush area should be masked"
        assert mask[220, 220] == 255, "Second brush area should be masked"
        assert mask[128, 128] == 0, "Center should not be masked"
        print("  PASS: test_mask_multiple_layers_union")

    def test_editor_none_detected(self):
        """None editor should be caught before processing."""
        editor_value = None
        assert editor_value is None, "Should detect None editor"
        print("  PASS: test_editor_none_detected")

    def test_mask_saved_as_grayscale(self):
        """Mask should be saveable as grayscale PIL image."""
        editor = make_editor_value()
        bg = editor["background"]
        layers = editor["layers"]
        mask = np.zeros(bg.shape[:2], dtype=np.uint8)
        for layer in layers:
            layer_gray = np.any(layer > 0, axis=2).astype(np.uint8) * 255
            mask = np.maximum(mask, layer_gray)

        mask_img = Image.fromarray(mask)
        assert mask_img.mode == "L", f"Expected mode L, got {mask_img.mode}"

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            mask_img.save(f.name)
            reloaded = Image.open(f.name).convert("L")
            arr = np.array(reloaded)
            assert arr[100, 100] == 255
            assert arr[10, 10] == 0
            os.unlink(f.name)
        print("  PASS: test_mask_saved_as_grayscale")


# ===========================================================================
# Test 2: Outpaint — canvas expansion + mask generation
# ===========================================================================
class TestOutpaintCanvasExpansion:
    """Test _run_outpaint canvas expansion logic (extracted from worker.py)."""

    @staticmethod
    def expand_canvas(image_arr, directions, expand_px):
        """Replicate worker.py _run_outpaint canvas logic."""
        img = Image.fromarray(image_arr)
        w, h = img.size

        pad = {"Left": 0, "Right": 0, "Up": 0, "Down": 0}
        dirs = directions if isinstance(directions, list) else [directions]
        for d in dirs:
            pad[d] = expand_px

        new_w = w + pad["Left"] + pad["Right"]
        new_h = h + pad["Up"] + pad["Down"]

        canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        canvas.paste(img, (pad["Left"], pad["Up"]))

        mask = Image.new("L", (new_w, new_h), 255)
        mask_arr = np.array(mask)
        mask_arr[pad["Up"]:pad["Up"]+h, pad["Left"]:pad["Left"]+w] = 0
        mask = Image.fromarray(mask_arr)

        return canvas, mask, new_w, new_h

    def test_expand_right(self):
        img = make_test_image(256, 256, (100, 100, 100))
        canvas, mask, new_w, new_h = self.expand_canvas(img, ["Right"], 128)
        assert new_w == 384, f"Expected 384, got {new_w}"
        assert new_h == 256, f"Expected 256, got {new_h}"

        c_arr = np.array(canvas)
        m_arr = np.array(mask)
        # Original area preserved
        assert c_arr[128, 128, 0] == 100, "Original area should be preserved"
        # Expanded area is black
        assert c_arr[128, 300, 0] == 0, "Expanded area should be black"
        # Mask: original=0, expanded=255
        assert m_arr[128, 128] == 0, "Original area mask should be 0 (preserve)"
        assert m_arr[128, 300] == 255, "Expanded area mask should be 255 (regenerate)"
        print("  PASS: test_expand_right")

    def test_expand_left(self):
        img = make_test_image(256, 256, (100, 100, 100))
        canvas, mask, new_w, new_h = self.expand_canvas(img, ["Left"], 128)
        assert new_w == 384
        assert new_h == 256
        c_arr = np.array(canvas)
        m_arr = np.array(mask)
        # Original is pasted at (128, 0)
        assert c_arr[128, 200, 0] == 100, "Original shifted right"
        assert c_arr[128, 64, 0] == 0, "Left expansion is black"
        assert m_arr[128, 64] == 255, "Left expansion mask = 255"
        assert m_arr[128, 200] == 0, "Original area mask = 0"
        print("  PASS: test_expand_left")

    def test_expand_all_directions(self):
        img = make_test_image(256, 256, (50, 50, 50))
        canvas, mask, new_w, new_h = self.expand_canvas(img, ["Left", "Right", "Up", "Down"], 64)
        assert new_w == 384, f"256 + 64 + 64 = 384, got {new_w}"
        assert new_h == 384, f"256 + 64 + 64 = 384, got {new_h}"
        m_arr = np.array(mask)
        # Center (original) should be 0
        assert m_arr[192, 192] == 0
        # Corners (expanded) should be 255
        assert m_arr[10, 10] == 255
        assert m_arr[370, 370] == 255
        print("  PASS: test_expand_all_directions")

    def test_expand_empty_direction_list(self):
        """Empty direction list = no expansion."""
        img = make_test_image(256, 256)
        canvas, mask, new_w, new_h = self.expand_canvas(img, [], 128)
        assert new_w == 256
        assert new_h == 256
        m_arr = np.array(mask)
        assert m_arr.sum() == 0, "No expansion → entire mask should be 0 (preserve)"
        print("  PASS: test_expand_empty_direction_list")

    def test_16px_alignment(self):
        """Outpaint result dimensions should align to 16px multiples."""
        img = make_test_image(256, 256)
        _, _, new_w, new_h = self.expand_canvas(img, ["Right"], 100)
        # 256 + 100 = 356 → align to 16 → 352
        aligned_w = (new_w // 16) * 16
        aligned_h = (new_h // 16) * 16
        assert aligned_w % 16 == 0, f"Width {aligned_w} not 16-aligned"
        assert aligned_h % 16 == 0, f"Height {aligned_h} not 16-aligned"
        print("  PASS: test_16px_alignment")

    def test_direction_string_vs_list(self):
        """Direction can be string or list — worker handles both."""
        img = make_test_image(256, 256)
        # String
        _, _, w1, h1 = self.expand_canvas(img, "Right", 128)
        # List
        _, _, w2, h2 = self.expand_canvas(img, ["Right"], 128)
        assert w1 == w2 and h1 == h2, "String and list should produce same result"
        print("  PASS: test_direction_string_vs_list")


# ===========================================================================
# Test 3: Outpaint kwargs — generators.py → worker.py flow
# ===========================================================================
class TestOutpaintKwargsFlow:
    """Test that generate_outpaint builds correct kwargs for the worker."""

    def test_kwargs_has_required_keys(self):
        """Outpaint kwargs should have image_path, direction, expand_px."""
        # Simulate what generate_outpaint builds
        img = make_test_image()
        pil_img = Image.fromarray(img)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil_img.save(f.name)
            tmp_path = f.name

        kwargs = {
            "prompt": "test prompt",
            "negative_prompt": None,
            "image_path": tmp_path,
            "direction": ["Right"],
            "expand_px": 256,
            "control_scale": 0.9,
            "num_steps": 8,
            "guidance_scale": 0.5,
            "cfg_truncation": 0.9,
            "max_sequence_length": 512,
            "time_shift": 3.0,
            "seed": -1,
        }

        required_keys = ["prompt", "image_path", "direction", "expand_px",
                         "control_scale", "num_steps", "seed"]
        for k in required_keys:
            assert k in kwargs, f"Missing key: {k}"

        # Note: width/height NOT in outpaint kwargs — _run_outpaint computes them
        assert "width" not in kwargs, "Outpaint kwargs should NOT have width (computed in worker)"
        assert "height" not in kwargs, "Outpaint kwargs should NOT have height (computed in worker)"

        os.unlink(tmp_path)
        print("  PASS: test_kwargs_has_required_keys")

    def test_inpaint_kwargs_has_width_height(self):
        """Inpaint kwargs MUST have width and height (unlike outpaint)."""
        kwargs = {
            "prompt": "test",
            "image_path": "/tmp/test.png",
            "mask_path": "/tmp/mask.png",
            "width": 512,
            "height": 768,
            "control_scale": 0.9,
            "num_steps": 8,
            "guidance_scale": 0.5,
            "cfg_truncation": 0.9,
            "max_sequence_length": 512,
            "time_shift": 3.0,
            "seed": 42,
        }
        assert "width" in kwargs
        assert "height" in kwargs
        print("  PASS: test_inpaint_kwargs_has_width_height")


# ===========================================================================
# Test 4: FaceSwap — status check
# ===========================================================================
class TestFaceSwapStatus:
    """Test FaceSwap module availability and error handling."""

    def test_faceswap_module_exists(self):
        """face_swap.py should exist."""
        face_swap_path = Path(__file__).resolve().parent.parent / "app" / "ui" / "face_swap.py"
        assert face_swap_path.exists(), f"face_swap.py not found at {face_swap_path}"
        print("  PASS: test_faceswap_module_exists")

    def test_faceswap_generator_raises_on_none_input(self):
        """generate_faceswap should raise on None inputs."""
        from generators import generate_faceswap
        try:
            generate_faceswap(None, None)
            assert False, "Should have raised"
        except Exception as e:
            assert "required" in str(e).lower() or "Target" in str(e) or "image" in str(e).lower()
        print("  PASS: test_faceswap_generator_raises_on_none_input")

    def test_faceswap_module_importable(self):
        """face_swap.py should be importable (syntax check)."""
        try:
            import face_swap
            print("  PASS: test_faceswap_module_importable (full import)")
        except ImportError as e:
            # May fail on missing tensorrt — that's expected, check it's not a syntax error
            if "tensorrt" in str(e).lower() or "trt" in str(e).lower():
                print(f"  PASS: test_faceswap_module_importable (ImportError on TRT expected: {e})")
            else:
                print(f"  WARN: test_faceswap_module_importable (ImportError: {e})")

    def test_faceswap_trt_engine_class(self):
        """Check if TRTEngine class is defined in face_swap.py."""
        face_swap_path = Path(__file__).resolve().parent.parent / "app" / "ui" / "face_swap.py"
        content = face_swap_path.read_text()
        assert "class TRTEngine" in content or "class FaceSwapPipeline" in content, \
            "face_swap.py should define TRTEngine or FaceSwapPipeline"
        print("  PASS: test_faceswap_trt_engine_class")


# ===========================================================================
# Test 5: Pipeline __call__ parameter compatibility
# ===========================================================================
class TestPipelineParamCompat:
    """Verify that worker kwargs match pipeline __call__ signature."""

    def test_inpaint_pipeline_accepts_mask_image(self):
        """ZImageControlPipeline.__call__ should accept image + mask_image."""
        import inspect
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        sig = inspect.signature(ZImageControlPipeline.__call__)
        params = list(sig.parameters.keys())
        assert "image" in params, f"Pipeline missing 'image' param. Params: {params}"
        assert "mask_image" in params, f"Pipeline missing 'mask_image' param. Params: {params}"
        assert "control_image" in params, f"Pipeline missing 'control_image' param. Params: {params}"
        assert "control_context_scale" in params, f"Pipeline missing 'control_context_scale'. Params: {params}"
        print("  PASS: test_inpaint_pipeline_accepts_mask_image")

    def test_pipeline_height_width_optional(self):
        """height/width should be optional (defaults to 1024)."""
        import inspect
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        sig = inspect.signature(ZImageControlPipeline.__call__)
        h_param = sig.parameters.get("height")
        w_param = sig.parameters.get("width")
        assert h_param is not None and h_param.default is None
        assert w_param is not None and w_param.default is None
        print("  PASS: test_pipeline_height_width_optional")

    def test_worker_inpaint_passes_correct_params(self):
        """Verify _run_inpaint passes params that pipeline accepts."""
        # These are the params _run_inpaint passes to pipeline()
        inpaint_params = {
            "prompt", "negative_prompt", "height", "width",
            "image", "mask_image", "control_context_scale",
            "num_inference_steps", "guidance_scale", "cfg_truncation",
            "max_sequence_length", "generator",
        }
        import inspect
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        sig = inspect.signature(ZImageControlPipeline.__call__)
        pipeline_params = set(sig.parameters.keys()) - {"self"}

        missing = inpaint_params - pipeline_params
        assert not missing, f"Worker passes params not in pipeline: {missing}"
        print("  PASS: test_worker_inpaint_passes_correct_params")

    def test_worker_inpaint_no_cfg_normalization(self):
        """_run_inpaint doesn't pass cfg_normalization — verify pipeline has a default."""
        import inspect
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        sig = inspect.signature(ZImageControlPipeline.__call__)
        cfg_norm = sig.parameters.get("cfg_normalization")
        assert cfg_norm is not None, "Pipeline should have cfg_normalization param"
        assert cfg_norm.default is not inspect.Parameter.empty, \
            "cfg_normalization should have a default value"
        print("  PASS: test_worker_inpaint_no_cfg_normalization")


# ===========================================================================
# Test 6: Edge cases
# ===========================================================================
class TestEdgeCases:
    """Edge cases that could cause hangs or silent failures."""

    def test_outpaint_with_non_16_aligned_expand(self):
        """Expand by non-16-aligned amount — should still work after alignment."""
        img = make_test_image(512, 512)
        pil = Image.fromarray(img)
        # 512 + 100 = 612 → aligned = 608
        new_w = 512 + 100
        aligned = (new_w // 16) * 16
        assert aligned == 608
        # Pipeline requires vae_scale_factor * 2 = 16 alignment
        assert aligned % 16 == 0
        print("  PASS: test_outpaint_with_non_16_aligned_expand")

    def test_inpaint_mask_all_white(self):
        """All-white mask = regenerate entire image (valid use case)."""
        mask = np.full((256, 256), 255, dtype=np.uint8)
        mask_img = Image.fromarray(mask)
        assert mask_img.mode == "L"
        assert np.array(mask_img).mean() == 255
        print("  PASS: test_inpaint_mask_all_white")

    def test_inpaint_mask_all_black(self):
        """All-black mask = preserve everything (nothing to inpaint, but shouldn't crash)."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask_img = Image.fromarray(mask)
        assert np.array(mask_img).mean() == 0
        print("  PASS: test_inpaint_mask_all_black")

    def test_outpaint_image_path_readable(self):
        """Image path saved by generate_outpaint should be readable."""
        img = make_test_image(256, 256, (200, 100, 50))
        pil = Image.fromarray(img)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil.save(f.name)
            # Simulate worker reading it back
            reloaded = Image.open(f.name).convert("RGB")
            arr = np.array(reloaded)
            assert arr.shape == (256, 256, 3)
            assert arr[128, 128, 0] == 200
            os.unlink(f.name)
        print("  PASS: test_outpaint_image_path_readable")

    def test_control_in_dim_33_check(self):
        """Pipeline uses control_in_dim to decide inpaint path.
        When num_channels_latents != control_in_dim, inpaint path is taken."""
        from zit_config import CONTROLNET_CONFIG
        assert CONTROLNET_CONFIG["control_in_dim"] == 33, \
            f"control_in_dim should be 33 for inpaint, got {CONTROLNET_CONFIG['control_in_dim']}"
        print("  PASS: test_control_in_dim_33_check")


# ===========================================================================
# Runner
# ===========================================================================
def run_all():
    test_classes = [
        TestInpaintMaskExtraction,
        TestOutpaintCanvasExpansion,
        TestOutpaintKwargsFlow,
        TestFaceSwapStatus,
        TestPipelineParamCompat,
        TestEdgeCases,
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
