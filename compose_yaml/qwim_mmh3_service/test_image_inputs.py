import base64
import io
import unittest
from PIL import Image
from image_inputs import decode_keyframe, video_size, MAX_VIDEO_PIXELS


def inline_image(width=935, height=1400, color='red'):
    buffer = io.BytesIO()
    Image.new('RGB', (width, height), color).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


class Keyframes(unittest.TestCase):
    def test_portrait_preserves_aspect_within_existing_pixel_budget(self):
        raw, info = decode_keyframe(inline_image())
        self.assertEqual((info['width'], info['height']), (935, 1400))
        self.assertEqual(video_size(935, 1400), (512, 768))
        self.assertEqual(len(info['sha256']), 64)
        for width, height in [(1920, 1080), (1080, 1920), (1024, 1024), (3000, 1000)]:
            w, h = video_size(width, height)
            self.assertLessEqual(w * h, MAX_VIDEO_PIXELS)
            self.assertEqual((w % 32, h % 32), (0, 0))

    def test_invalid_remote_mislabeled_and_oversized_inputs_are_rejected(self):
        for value in ['https://example.com/image.png', 'data:image/png;base64,invalid!',
                      inline_image(2, 2).replace('image/png', 'image/jpeg'),
                      inline_image(4097, 4097)]:
            with self.assertRaises(ValueError):
                decode_keyframe(value)

    def test_exif_rotation_controls_output_orientation(self):
        buffer = io.BytesIO(); image = Image.new('RGB', (1400, 935))
        exif = Image.Exif(); exif[274] = 6
        image.save(buffer, format='JPEG', exif=exif)
        _, info = decode_keyframe('data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode())
        self.assertEqual(video_size(info['width'], info['height']), (512, 768))
