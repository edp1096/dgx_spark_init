"""Read timestamped WebM recordings when Decord cannot infer frame metadata.

PyAV scans decoded frame metadata once, then decodes only requested RGB frames
into memory. The original encoded video is never rewritten or transcoded.
"""
import logging
import math
import numpy as np


class _Batch:
    def __init__(self, value):
        self.value = value

    def asnumpy(self):
        return self.value


class PyAVReader:
    def __init__(self, source):
        import av
        self.source = source
        self.count = 0
        first = last = None
        with av.open(source) as container:
            if not container.streams.video:
                raise ValueError("Video has no video stream")
            stream = container.streams.video[0]
            stream.thread_count = 2
            duration = float(container.duration / av.time_base) if container.duration else 0.0
            rate = float(stream.average_rate) if stream.average_rate else 0.0
            for frame in container.decode(stream):
                self.count += 1
                if frame.time is not None:
                    timestamp = float(frame.time)
                    first = timestamp if first is None else first
                    last = timestamp
            if not self.count:
                raise ValueError("Video has no decodable frames")
            if duration <= 0 and first is not None and last > first:
                duration = (last - first) * self.count / max(1, self.count - 1)
            self.fps = self.count / duration if duration > 0 else rate
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("Video has no usable timestamps or frame rate")

    def __len__(self):
        return self.count

    def get_avg_fps(self):
        return self.fps

    def __getitem__(self, index):
        return _Batch(self.get_batch([index]).asnumpy()[0])

    def get_batch(self, indices):
        import av
        indices = [int(i) for i in indices]
        if not indices or any(i < 0 or i >= self.count for i in indices):
            raise IndexError("Video frame index out of range")
        wanted = set(indices)
        frames = {}
        with av.open(self.source) as container:
            stream = container.streams.video[0]
            stream.thread_count = 2
            for i, frame in enumerate(container.decode(stream)):
                if i in wanted:
                    frames[i] = frame.to_ndarray(format="rgb24")
                    if len(frames) == len(wanted):
                        break
        if len(frames) != len(wanted):
            raise ValueError("Video ended before requested frames could be decoded")
        return _Batch(np.stack([frames[i] for i in indices]))


def read_decord_or_pyav(source, ctx):
    from decord import DECORDError, VideoReader
    try:
        return VideoReader(source, ctx=ctx)
    except (DECORDError, RuntimeError) as exc:
        if "Failed to measure duration/frame-count" not in str(exc):
            raise
        logging.getLogger(__name__).warning(
            "Decord could not read video metadata; decoding original video with PyAV."
        )
        return PyAVReader(source)
