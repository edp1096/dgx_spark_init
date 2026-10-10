"""Validate inline keyframes and choose an aspect-matched, bounded canvas."""
import base64
import binascii
import hashlib
import io
import math

from PIL import Image

MAX_IMAGE_BYTES = 32 * 1024**2
MAX_IMAGE_PIXELS = 16 * 1024**2
MAX_VIDEO_PIXELS = 864 * 480


def decode_keyframe(value):
    header, separator, encoded = value.partition(',')
    formats = {'data:image/png;base64': 'PNG', 'data:image/jpeg;base64': 'JPEG',
               'data:image/webp;base64': 'WEBP'}
    if not separator or header not in formats or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError('Keyframe must be a PNG/JPEG/WebP data URL of at most 32 MiB')
    try:
        raw = base64.b64decode(encoded, validate=True)
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError('Keyframe exceeds 32 MiB or is empty')
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError('Keyframe must have at most 16 megapixels')
            if image.format != formats[header]:
                raise ValueError('Keyframe image does not match its declared format')
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            if getattr(image, 'is_animated', False):
                raise ValueError('Keyframe must be a static image')
            if image.getexif().get(274, 1) in (5, 6, 7, 8):
                width, height = height, width
    except (binascii.Error, OSError, Image.DecompressionBombError) as error:
        raise ValueError('Invalid keyframe image') from error
    return raw, {'width': width, 'height': height, 'sha256': hashlib.sha256(raw).hexdigest()}


def video_size(width, height):
    ratio = width / height
    candidates = [(abs(math.log((w / h) / ratio)), w * h, w, h)
                  for w in range(128, 1025, 32) for h in range(128, 1025, 32)
                  if w * h <= MAX_VIDEO_PIXELS]
    closest = min(row[0] for row in candidates)
    # Allow <= about 1% aspect error beyond the closest grid fit, then maximize
    # detail without exceeding the already-qualified landscape pixel budget.
    _, _, width, height = max((row for row in candidates if row[0] <= closest + .01),
                             key=lambda row: (row[1], -row[0]))
    return width, height
