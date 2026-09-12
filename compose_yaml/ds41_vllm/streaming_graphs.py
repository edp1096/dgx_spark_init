"""Breakable full-attention graphs with mandatory CPU expert I/O breaks.

The pinned vLLM's ordinary eager-break decorator deliberately ignores FULL
mode so attention is captured. Expert I/O must still break in that mode.
"""
from functools import wraps

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    _weak_ref_capture_arg,
)

# Only set while recording the bounded-context variant. Runtime selection
# uses the runner's CPU upper bound, which can overestimate but cannot omit
# live tokens. The normal graph handles every other request.
SHORT_CONTEXT_LIMIT = 512
capture_context_limit = None
runtime_context_upper_bound = float('inf')


def can_capture_all_index_candidates(compress_ratio, topk_tokens):
    return (capture_context_limit is not None
            and capture_context_limit // compress_ratio <= topk_tokens)


def expert_io_break(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        capture = BreakableCUDAGraphCapture.current()
        if capture is None or not capture._capturing:
            return fn(*args, **kwargs)
        weak_args = tuple(_weak_ref_capture_arg(arg) for arg in args)
        weak_kwargs = {key: _weak_ref_capture_arg(arg) for key, arg in kwargs.items()}
        return capture.add_eager(lambda: fn(*weak_args, **weak_kwargs))

    return wrapper
