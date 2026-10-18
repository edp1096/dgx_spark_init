"""Numerical checks for mixed FP8 formats and conservative delta transfer."""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numeric import candidates, dequantize, mapped, q8_decode, requantize, transform


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def matrix(self, kind):
        torch.manual_seed(17)
        weight = (torch.randn(256, 256) * 3).to(torch.float8_e4m3fn)
        if kind == "mxfp8":
            scale = torch.randint(120, 126, (256, 8), dtype=torch.uint8)
        else:
            scale = torch.tensor([0.125, 0.25, 0.5, 1.0]).reshape(2, 1, 2, 1)
        return weight, scale

    def test_q8_layout_and_values(self):
        blocks = np.zeros(4, dtype=np.dtype([("d", "<f2"), ("q", "i1", (32,))]))
        for i, scale in enumerate((0.5, 1.0, 2.0, 4.0)):
            blocks[i]["d"] = scale
            blocks[i]["q"] = np.arange(-16, 16, dtype=np.int8)
        result = q8_decode(blocks.tobytes(), 2, 64)
        expected = np.concatenate([np.arange(-16, 16) * x for x in (0.5, 1.0, 2.0, 4.0)]).reshape(2, 64)
        np.testing.assert_array_equal(result.numpy(), expected)
        with self.assertRaises(ValueError):
            q8_decode(blocks.tobytes()[:-1], 2, 64)

    def test_fp8_block_scale_axes(self):
        weight = torch.ones(256, 256).to(torch.float8_e4m3fn)
        scales = torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(2, 1, 2, 1)
        result, kind = dequantize(weight, scales)
        self.assertEqual(kind, "fp8_pb_128")
        for row, column, expected in [(0, 0, 1), (0, 1, 2), (1, 0, 4), (1, 1, 8)]:
            self.assertTrue(torch.all(result[row * 128:(row + 1) * 128, column * 128:(column + 1) * 128] == expected))

    def test_mxfp8_exponent_bias_and_axes(self):
        weight = torch.ones(2, 64).to(torch.float8_e4m3fn)
        scales = torch.tensor([[127, 128], [126, 125]], dtype=torch.uint8)
        result, kind = dequantize(weight, scales)
        self.assertEqual(kind, "mxfp8")
        for row, column, expected in [(0, 0, 1), (0, 1, 2), (1, 0, 0.5), (1, 1, 0.25)]:
            self.assertTrue(torch.all(result[row, column * 32:(column + 1) * 32] == expected))

    def test_zero_delta_keeps_exact_source_bytes(self):
        for kind in ("mxfp8", "fp8_pb_128"):
            with self.subTest(kind=kind):
                weight, scale = self.matrix(kind)
                base, _ = dequantize(weight, scale)
                q, s, restored, counts = requantize(base, weight, scale)
                self.assertTrue(torch.equal(q.view(torch.uint8), weight.view(torch.uint8)))
                self.assertTrue(torch.equal(s, scale))
                self.assertTrue(torch.equal(restored, base))
                self.assertEqual(counts["original_scale_blocks"] + counts["fresh_scale_blocks"], 0)

    def test_each_block_chooses_minimum_reconstruction_error(self):
        for kind in ("mxfp8", "fp8_pb_128"):
            with self.subTest(kind=kind):
                weight, scale = self.matrix(kind)
                base, _ = dequantize(weight, scale)
                torch.manual_seed(21)
                target = base + torch.randn_like(base) * 0.35
                _, choices = candidates(target, weight, scale)
                q, s, result, _ = requantize(target, weight, scale)
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(q.shape, weight.shape)
                self.assertEqual(s.shape, scale.shape)
                if kind == "mxfp8":
                    # An independent scalar block loop checks the chosen reconstruction.
                    blocks = [(r, r + 1, c, c + 32) for r in (0, 19, 255) for c in (0, 96, 224)]
                else:
                    blocks = [(r, r + 128, c, c + 128) for r in (0, 128) for c in (0, 128)]
                for r0, r1, c0, c1 in blocks:
                    expected = min(float((x[r0:r1, c0:c1].double() - target[r0:r1, c0:c1].double()).square().sum())
                                   for _, _, x in choices)
                    actual = float((result[r0:r1, c0:c1].double() - target[r0:r1, c0:c1].double()).square().sum())
                    self.assertAlmostEqual(actual, expected, places=9)

    def test_nonzero_delta_improves_target_error(self):
        for kind in ("mxfp8", "fp8_pb_128"):
            with self.subTest(kind=kind):
                weight, scale = self.matrix(kind)
                base, _ = dequantize(weight, scale)
                original = base.clone()
                donor = original + 0.25
                _, _, resolved, stats = transform(original, donor, weight, scale)
                self.assertEqual(resolved, kind)
                self.assertGreater(stats["effect_norm2"], 0)
                self.assertGreater(stats["delta_dot_effect"], 0)
                self.assertLessEqual(stats["error_norm2"], stats["delta_norm2"] + 1e-9)

    def test_declared_projection_mapping(self):
        for layer in range(15, 37):
            for suffix in ("attn_output", "ffn_down_shexp"):
                columns = 2048 if suffix == "ffn_down_shexp" else (16384 if layer % 4 == 3 else 8192)
                row = {"name": f"blk.{layer}.{suffix}.weight", "type": 8,
                       "shape": [columns, 4096], "bytes": columns * 4096 // 32 * 34}
                prefix, shape, kind = mapped(row)
                self.assertEqual(shape, [4096, columns])
                self.assertNotIn(".experts.", prefix)
                self.assertEqual(kind, "mxfp8" if suffix == "attn_output" and layer % 4 != 3 else "fp8_pb_128")
        with self.assertRaises(ValueError):
            mapped({"name": "blk.20.ffn_down_exps.weight", "type": 8, "shape": [2048, 4096, 288], "bytes": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
