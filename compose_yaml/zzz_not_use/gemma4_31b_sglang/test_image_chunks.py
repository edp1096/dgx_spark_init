import types
import unittest

from patch_image_chunks import HELPER

scope = {}
exec(HELPER, scope)
chunk_length = scope['_gemma_image_chunk_length']


class ImageChunksTest(unittest.TestCase):
    def request(self, start=0, offsets=((400, 679),)):
        item = types.SimpleNamespace(is_image=lambda: True, offsets=offsets)
        return types.SimpleNamespace(
            prefix_indices=[0] * start,
            multimodal_inputs=types.SimpleNamespace(mm_items=[item]))

    def test_cuts_before_image_then_processes_whole_image(self):
        self.assertEqual(chunk_length(self.request(), 512, 1), 400)
        self.assertEqual(chunk_length(self.request(400), 512, 1), 512)

    def test_alignment_and_multiple_images(self):
        self.assertEqual(chunk_length(self.request(200), 400, 16), 200)
        self.assertEqual(chunk_length(self.request(offsets=((50, 329), (400, 679))), 512, 1), 400)

    def test_rejects_partial_cached_image(self):
        with self.assertRaises(RuntimeError):
            chunk_length(self.request(500), 512, 1)

    def test_rejects_image_larger_than_chunk(self):
        with self.assertRaises(RuntimeError):
            chunk_length(self.request(400), 128, 1)

    def test_text_only_unchanged(self):
        self.assertEqual(chunk_length(types.SimpleNamespace(multimodal_inputs=None), 512, 1), 512)


if __name__ == '__main__':
    unittest.main()
