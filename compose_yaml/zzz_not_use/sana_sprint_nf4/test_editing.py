import unittest

from PIL import Image

from editing import build_outpaint_canvas


class OutpaintCanvasTest(unittest.TestCase):
    def test_places_source_and_masks_only_expansion_without_overlap(self):
        source = Image.new("RGB", (256, 256), "red")
        result = build_outpaint_canvas(
            source, left=64, right=128, top=0, bottom=64, overlap=0, feather=0
        )

        self.assertEqual(result.image.size, (448, 320))
        self.assertEqual(result.image.getpixel((64, 0)), (255, 0, 0))
        self.assertEqual(result.mask.getpixel((64, 0)), 0)
        self.assertEqual(result.mask.getpixel((10, 10)), 255)
        self.assertEqual(result.mask.getpixel((400, 100)), 255)

    def test_overlap_moves_mask_inside_extended_source_edges(self):
        source = Image.new("RGB", (256, 256), "blue")
        result = build_outpaint_canvas(
            source, left=64, right=64, top=64, bottom=64, overlap=32, feather=0
        )

        self.assertEqual(result.mask.getpixel((70, 192)), 255)
        self.assertEqual(result.mask.getpixel((110, 192)), 0)
        self.assertEqual(result.mask.getpixel((300, 192)), 255)

    def test_rejects_overlap_that_consumes_the_source(self):
        source = Image.new("RGB", (256, 256), "blue")
        with self.assertRaises(ValueError):
            build_outpaint_canvas(
                source, left=64, right=64, top=0, bottom=0, overlap=128, feather=0
            )


if __name__ == "__main__":
    unittest.main()
