import unittest
import numpy as np
import gguf
import gguf.quants
from converter import streaming_rows


class ConversionMemoryTests(unittest.TestCase):
    def test_streaming_matches_upstream_bytes(self):
        original = gguf.quants._apply_over_grouped_rows
        rng = np.random.default_rng(47)
        for rows in [1, 15, 16, 17, 33, 80]:
            source = rng.normal(size=(rows, 256)).astype(np.float32)
            source[0, :7] = [0., -0., np.inf, -np.inf, np.nan, 1e-38, -1e-38]
            for qtype in [gguf.GGMLQuantizationType.BF16]:
                expected = gguf.quants.quantize(source, qtype)
                try:
                    gguf.quants._apply_over_grouped_rows = streaming_rows
                    actual = gguf.quants.quantize(source, qtype)
                finally:
                    gguf.quants._apply_over_grouped_rows = original
                np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
