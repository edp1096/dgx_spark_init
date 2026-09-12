"""Diagnostics for exact dependency-preserving final-layer routed-expert row selection.

No weights, attention edges, cache boundaries or draft context are omitted.
Only rows of the final output without an inference consumer may be omitted.
"""
import json
import os


def validate_enabled():
    if os.environ.get('DSV41_BENCH_CONTROL') != '1':
        return False
    import cache_control
    return cache_control.validate_final_decoder


def record_selection(total, selected, reference, actual):
    row = {'input_rows': total, 'output_rows': selected, 'layer': 39, 'scope': 'routed_experts_only'}
    if reference is not None:
        import torch
        row['validation'] = []
        for expected, value in zip(reference, actual, strict=True):
            delta = (expected.float() - value.float()).abs()
            row['validation'].append({'max_abs': delta.max().item(),
                                      'rms': delta.square().mean().sqrt().item(),
                                      'within_tolerance': bool(torch.allclose(value, expected, atol=0.015625, rtol=0.01))})
            # Shape-dependent FP8 GEMM reductions may differ; no extra
            # quantization or truncated attention is allowed.
    print('FINAL_DECODER_ROWS ' + json.dumps(row), flush=True)
