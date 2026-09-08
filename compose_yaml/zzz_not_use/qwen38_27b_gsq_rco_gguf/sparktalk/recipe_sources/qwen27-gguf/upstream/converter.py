#!/usr/bin/env python3
"""Run the pinned converter with bounded temporary row-quantization storage."""
from pathlib import Path
import runpy
import sys
import numpy as np
import gguf.quants


def streaming_rows(func, arr, otype, oshape):
    rows = arr.reshape((-1, arr.shape[-1]))
    out = np.empty(shape=int(np.prod(oshape)), dtype=otype)
    # Same row grouping as upstream, but release each result immediately.
    # Upstream retains every group until concatenate, duplicating a full tensor.
    groups = (rows.shape[0] // 16) or 1
    offset = 0
    for group in np.array_split(rows, groups):
        result = func(group).ravel()
        out[offset:offset + result.size] = result
        offset += result.size
    if offset != out.size:
        raise ValueError('Quantized row size does not match output shape')
    return out.reshape(oshape)


if __name__ == '__main__':
    gguf.quants._apply_over_grouped_rows = streaming_rows
    source = Path('/opt/llama.cpp')
    sys.path.insert(0, str(source))
    runpy.run_path(str(source / 'convert_hf_to_gguf.py'), run_name='__main__')
