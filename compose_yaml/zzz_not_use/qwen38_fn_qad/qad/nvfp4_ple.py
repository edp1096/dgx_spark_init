"""NVFP4 PLE row decoding, independently implemented from E2M1/block-scale format.

Packed checkpoint bytes stay unchanged. The loader lives in ple_embedding.py.
The global FP32 scale is applied before BF16 output rounding.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather(weight, scales, global_scale, ids, output,
            START: tl.constexpr, END: tl.constexpr, D: tl.constexpr,
            B: tl.constexpr):
    row = tl.program_id(0)
    index = tl.load(ids + row)
    valid = (index >= START) & (index < END)
    local = tl.where(valid, index - START, 0)
    col = tl.arange(0, B)
    mask = (col < D) & valid
    # Integer addresses permit the existing SGLang pinned/file-backed host path.
    w = weight.to(tl.int64).to(tl.pointer_type(tl.uint8))
    s = scales.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    packed = tl.load(w + local * (D // 2) + col // 2, mask, 0)
    bits = (packed >> ((col % 2) * 4)) & 15
    magnitude = bits & 7
    value = tl.where(magnitude < 2, magnitude * 0.5,
            tl.where(magnitude < 4, 1.0 + (magnitude - 2) * 0.5,
            tl.where(magnitude < 6, 2.0 + (magnitude - 4),
                     4.0 + (magnitude - 6) * 2.0)))
    value = tl.where((bits & 8) != 0, -value, value)
    block_scale = tl.load(s + local * (D // 16) + col // 16, mask, 0.0).to(tl.float32)
    factor = tl.load(global_scale).to(tl.float32)
    # Match ModelOpt: reconstruct the FP32 block scale before multiplying E2M1.
    value = value * (block_scale * factor)
    tl.store(output + row * D + col, tl.where(valid, value, 0), col < D)


def gather(weight, scales, global_scale, ids, start=0, out=None):
    """Gather rows from packed GPU or CUDA-registered CPU tensors."""
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise ValueError('weight must be a packed uint8 matrix')
    dim = weight.shape[1] * 2
    if dim % 16 or scales.shape != (weight.shape[0], dim // 16):
        raise ValueError('one E4M3 scale per 16 unpacked columns required')
    if scales.dtype != torch.float8_e4m3fn:
        raise ValueError('scales must be E4M3')
    if ids.device.type != 'cuda' or ids.dtype != torch.int64 or not ids.is_contiguous():
        raise ValueError('ids must be contiguous CUDA int64')
    if global_scale.device != ids.device or global_scale.dtype != torch.float32 or global_scale.numel()!=1:
        raise ValueError('global scale must be one CUDA FP32 value')
    if not weight.is_contiguous() or not scales.is_contiguous():
        raise ValueError('row-major contiguous checkpoint tensors required')
    shape = (*ids.shape, dim)
    if out is None:
        out = torch.empty(shape, device=ids.device, dtype=torch.bfloat16)
    if out.shape != shape or out.dtype != torch.bfloat16 or out.device != ids.device or not out.is_contiguous():
        raise ValueError('invalid output buffer')
    if ids.numel():
        _gather[(ids.numel(),)](weight.data_ptr(), scales.data_ptr(), global_scale,
                              ids, out, start, start+weight.shape[0], dim,
                              triton.next_power_of_2(dim))
    return out
