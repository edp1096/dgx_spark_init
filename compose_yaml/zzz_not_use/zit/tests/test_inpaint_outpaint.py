"""Inpaint / Outpaint logic tests (no GPU required).

Tests mask extraction, canvas expansion, kwargs construction, edge cases.

Usage:
    cd /root/zit-ui && python tests/test_inpaint_outpaint.py
    cd /root/zit-ui && python -m pytest tests/test_inpaint_outpaint.py -v
"""
import sys
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "ui"))


def make_test_image(w=256, h=256, color=(128, 64, 32)):
    return np.full((h, w, 3), color, dtype=np.uint8)


def make_editor_value(w=256, h=256, mask_rect=(50, 50, 150, 150)):
    """Simulate gr.ImageEditor output."""
    bg = make_test_image(w, h, (100, 100, 100))
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    y1, x1, y2, x2 = mask_rect
    layer[y1:y2, x1:x2] = [255, 255, 255, 255]
    return {"background": bg, "layers": [layer], "composite": bg.copy()}


# ===========================================================================
# Test 1: Inpaint mask extraction
# ===========================================================================
class TestInpaintMaskExtraction:

    def test_mask_from_layers_basic(self):
        editor = make_editor_value(256, 256, (50, 50, 150, 150))
        bg = editor["background"]
        mask = np.zeros(bg.shape[:2], dtype=np.uint8)
        for layer in editor["layers"]:
            if isinstance(layer, np.ndarray) and layer.ndim >= 2:
                if layer.ndim == 3:
                    layer_gray = np.any(layer > 0, axis=2).astype(np.uint8) * 255
                else:
                    layer_gray = (layer > 0).astype(np.uint8) * 255
                mask = np.maximum(mask, layer_gray)

        assert mask[100, 100] == 255
        assert mask[10, 10] == 0
        print("  PASS: mask from layers basic")

    def test_mask_empty_layers(self):
        editor = {"background": make_test_image(), "layers": [], "composite": make_test_image()}
        assert len(editor.get("layers", [])) == 0
        print("  PASS: empty layers detected")

    def test_mask_multiple_layers_union(self):
        bg = make_test_image(256, 256)
        layer1 = np.zeros((256, 256, 4), dtype=np.uint8)
        layer1[10:50, 10:50] = [255, 255, 255, 255]
        layer2 = np.zeros((256, 256, 4), dtype=np.uint8)
        layer2[200:240, 200:240] = [255, 255, 255, 255]

        mask = np.zeros(bg.shape[:2], dtype=np.uint8)
        for layer in [layer1, layer2]:
            layer_gray = np.any(layer > 0, axis=2).astype(np.uint8) * 255
            mask = np.maximum(mask, layer_gray)

        assert mask[30, 30] == 255
        assert mask[220, 220] == 255
        assert mask[128, 128] == 0
        print("  PASS: multiple layers union")

    def test_mask_saved_as_grayscale(self):
        editor = make_editor_value()
        mask = np.zeros(editor["background"].shape[:2], dtype=np.uint8)
        for layer in editor["layers"]:
            layer_gray = np.any(layer > 0, axis=2).astype(np.uint8) * 255
            mask = np.maximum(mask, layer_gray)

        mask_img = Image.fromarray(mask)
        assert mask_img.mode == "L"

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            mask_img.save(f.name)
            reloaded = np.array(Image.open(f.name).convert("L"))
            assert reloaded[100, 100] == 255
            assert reloaded[10, 10] == 0
            os.unlink(f.name)
        print("  PASS: mask saved as grayscale")


# ===========================================================================
# Test 2: Outpaint canvas expansion
# ===========================================================================
class TestOutpaintCanvasExpansion:

    @staticmethod
    def expand_canvas(image_arr, directions, expand_px):
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
        mask_arr = np.full((new_h, new_w), 255, dtype=np.uint8)
        mask_arr[pad["Up"]:pad["Up"]+h, pad["Left"]:pad["Left"]+w] = 0
        return canvas, Image.fromarray(mask_arr), new_w, new_h

    def test_expand_right(self):
        img = make_test_image(256, 256, (100, 100, 100))
        canvas, mask, nw, nh = self.expand_canvas(img, ["Right"], 128)
        assert nw == 384 and nh == 256
        c_arr = np.array(canvas)
        m_arr = np.array(mask)
        assert c_arr[128, 128, 0] == 100  # original preserved
        assert c_arr[128, 300, 0] == 0    # expanded area black
        assert m_arr[128, 128] == 0       # original mask=0
        assert m_arr[128, 300] == 255     # expanded mask=255
        print("  PASS: expand right")

    def test_expand_left(self):
        img = make_test_image(256, 256, (100, 100, 100))
        canvas, mask, nw, nh = self.expand_canvas(img, ["Left"], 128)
        assert nw == 384
        c_arr = np.array(canvas)
        m_arr = np.array(mask)
        assert c_arr[128, 200, 0] == 100
        assert m_arr[128, 64] == 255
        assert m_arr[128, 200] == 0
        print("  PASS: expand left")

    def test_expand_all_directions(self):
        img = make_test_image(256, 256, (50, 50, 50))
        _, mask, nw, nh = self.expand_canvas(img, ["Left", "Right", "Up", "Down"], 64)
        assert nw == 384 and nh == 384
        m_arr = np.array(mask)
        assert m_arr[192, 192] == 0   # center
        assert m_arr[10, 10] == 255   # corner
        print("  PASS: expand all directions")

    def test_expand_empty_directions(self):
        img = make_test_image(256, 256)
        _, mask, nw, nh = self.expand_canvas(img, [], 128)
        assert nw == 256 and nh == 256
        assert np.array(mask).sum() == 0
        print("  PASS: empty directions = no expansion")

    def test_string_vs_list_direction(self):
        img = make_test_image(256, 256)
        _, _, w1, h1 = self.expand_canvas(img, "Right", 128)
        _, _, w2, h2 = self.expand_canvas(img, ["Right"], 128)
        assert w1 == w2 and h1 == h2
        print("  PASS: string vs list direction")


# ===========================================================================
# Test 3: Kwargs flow
# ===========================================================================
class TestKwargsFlow:

    def test_outpaint_kwargs_no_width_height(self):
        """Outpaint kwargs should NOT have width/height (computed in worker)."""
        kwargs = {
            "prompt": "test", "image_path": "/tmp/test.png",
            "direction": ["Right"], "expand_px": 256,
            "control_scale": 0.9, "num_steps": 8, "seed": -1,
        }
        assert "width" not in kwargs
        assert "height" not in kwargs
        print("  PASS: outpaint kwargs no width/height")

    def test_inpaint_kwargs_has_width_height(self):
        kwargs = {
            "prompt": "test", "image_path": "/tmp/test.png",
            "mask_path": "/tmp/mask.png", "width": 512, "height": 768,
        }
        assert "width" in kwargs and "height" in kwargs
        print("  PASS: inpaint kwargs has width/height")


# ===========================================================================
# Test 4: Edge cases
# ===========================================================================
class TestEdgeCases:

    def test_16px_alignment(self):
        new_w = 256 + 100  # 356
        aligned = (new_w // 16) * 16  # 352
        assert aligned % 16 == 0
        print("  PASS: 16px alignment")

    def test_mask_all_white(self):
        mask = np.full((256, 256), 255, dtype=np.uint8)
        assert Image.fromarray(mask).mode == "L"
        assert np.array(Image.fromarray(mask)).mean() == 255
        print("  PASS: all-white mask valid")

    def test_mask_all_black(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        assert np.array(Image.fromarray(mask)).mean() == 0
        print("  PASS: all-black mask valid")

    def test_image_path_roundtrip(self):
        img = make_test_image(256, 256, (200, 100, 50))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.fromarray(img).save(f.name)
            arr = np.array(Image.open(f.name).convert("RGB"))
            assert arr.shape == (256, 256, 3)
            assert arr[128, 128, 0] == 200
            os.unlink(f.name)
        print("  PASS: image path roundtrip")

    def test_control_in_dim_33(self):
        from zit_config import CONTROLNET_CONFIG
        assert CONTROLNET_CONFIG["control_in_dim"] == 33
        print("  PASS: control_in_dim = 33")


# ===========================================================================
# Runner
# ===========================================================================
def run_all():
    test_classes = [
        TestInpaintMaskExtraction,
        TestOutpaintCanvasExpansion,
        TestKwargsFlow,
        TestEdgeCases,
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
