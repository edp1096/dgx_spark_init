"""Regression tests for the narrow Decord metadata fallback."""
import unittest
from unittest.mock import patch
import numpy as np
from decord import DECORDError
from sglang.srt.utils.video_fallback import PyAVReader, read_decord_or_pyav


class Frame:
    def __init__(self, n):
        self.time = n * 0.5
        self.n = n

    def to_ndarray(self, format):
        assert format == 'rgb24'
        return np.full((2, 2, 3), self.n, dtype=np.uint8)


class Container:
    duration = None

    def __init__(self, frames):
        self.frames = frames
        self.streams = type('Streams', (), {'video': [type('Stream', (), {'average_rate': None})()]})()

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def decode(self, stream): return iter(self.frames)


class VideoFallbackTests(unittest.TestCase):
    def test_regular_decord_is_unchanged(self):
        expected = object()
        with patch('decord.VideoReader', return_value=expected), patch('sglang.srt.utils.video_fallback.PyAVReader') as fallback:
            self.assertIs(read_decord_or_pyav('clip.webm', 'cpu'), expected)
            fallback.assert_not_called()

    def test_only_metadata_errors_fall_back(self):
        with patch('decord.VideoReader', side_effect=DECORDError('Failed to measure duration/frame-count due to broken metadata.')), patch('sglang.srt.utils.video_fallback.PyAVReader', return_value='decoded'):
            self.assertEqual(read_decord_or_pyav('clip.webm', 'cpu'), 'decoded')
        for error in [DECORDError('corrupt bitstream'), MemoryError('OOM')]:
            with patch('decord.VideoReader', side_effect=error), patch('sglang.srt.utils.video_fallback.PyAVReader') as fallback:
                with self.assertRaises(type(error)): read_decord_or_pyav('clip.webm', 'cpu')
                fallback.assert_not_called()

    def test_missing_container_metadata_and_frame_selection(self):
        frames = [Frame(i) for i in range(4)]
        with patch('av.open', side_effect=lambda _: Container(frames)):
            reader = PyAVReader('clip.webm')
            self.assertEqual(len(reader), 4)
            self.assertEqual(reader.get_avg_fps(), 2)
            batch = reader.get_batch([3, 0, 3, 1]).asnumpy()
            self.assertEqual(batch[:, 0, 0, 0].tolist(), [3, 0, 3, 1])
            with self.assertRaises(IndexError): reader.get_batch([4])
        with patch('av.open', return_value=Container([])):
            with self.assertRaisesRegex(ValueError, 'no decodable frames'): PyAVReader('empty.webm')


if __name__ == '__main__': unittest.main()
