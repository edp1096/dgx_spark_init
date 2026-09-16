"""CPU reference encoders for unswizzled ModelOpt checkpoint tensors.

These perform weight-only rounding, not activation calibration, AWQ or GPTQ.
FP4 packing follows vendored ModelOpt NVFP4QTensor: low nibble first,
16-element blocks, E4M3 block scales and one FP32 global scale.
"""
import numpy as np


def e4m3_values():
    codes = np.arange(127, dtype=np.uint8)  # 0x7f is NaN, 0x7e is 448
    exp = (codes >> 3).astype(np.int32)
    mant = (codes & 7).astype(np.float32)
    return np.where(exp == 0, mant * 2.0**-9,
                    (1 + mant / 8) * np.exp2(exp - 7)).astype(np.float32)


def nearest(values, table):
    hi = np.searchsorted(table, values).clip(0, len(table) - 1)
    lo = (hi - 1).clip(0)
    left, right = values - table[lo], table[hi] - values
    # Round halfway to even significand (even encoded index).
    use_lo = (left < right) | ((left == right) & ((lo & 1) == 0))
    return np.where(use_lo, lo, hi).astype(np.uint8)


def fp8_encode(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite FP8 input')
    codes = nearest(np.abs(values).clip(0, 448), e4m3_values())
    return codes | (np.signbit(values).astype(np.uint8) << 7)


def fp8_decode(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    if np.any((codes & 127) == 127):
        raise ValueError('FP8 NaN')
    return e4m3_values()[codes & 127] * np.where(codes & 128, -1, 1)


def quantize_fp8(values, scale_shape, block_size=None, min_scale=1e-30):
    """Per-tensor or per-output-channel E4M3. Unknown layouts are rejected."""
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all() or values.ndim != 2:
        raise ValueError('Expected finite 2D weight')
    if block_size is not None:
        br, bc = block_size
        rows, cols = values.shape
        nr, nc = (rows+br-1)//br, (cols+bc-1)//bc
        padded = np.pad(values, ((0,nr*br-rows),(0,nc*bc-cols)))
        blocks = padded.reshape(nr,br,nc,bc).transpose(0,2,1,3)
        scale = np.maximum(np.max(np.abs(blocks),axis=(2,3))/448,min_scale).astype(np.float32)
        expanded = np.repeat(np.repeat(scale,br,axis=0),bc,axis=1)[:rows,:cols]
    elif np.prod(scale_shape, dtype=np.int64) == 1:
        scale = np.array(max(float(np.max(np.abs(values))) / 448, min_scale), dtype=np.float32)
        expanded = scale
    elif list(scale_shape) in ([values.shape[0]], [values.shape[0], 1]):
        scale = np.maximum(np.max(np.abs(values), axis=1) / 448, min_scale).astype(np.float32)
        expanded = scale[:, None]
    else:
        raise ValueError(f'Unsupported FP8 scale layout: {scale_shape}')
    codes = fp8_encode(values / expanded)
    reconstructed = fp8_decode(codes) * expanded
    return codes, scale.reshape(scale_shape), reconstructed


def quantize_nvfp4(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 16 or not np.isfinite(values).all():
        raise ValueError('NVFP4 requires finite 2D weights with columns divisible by 16')
    amax = float(np.max(np.abs(values)))
    global_scale = np.float32(amax / (6 * 448) if amax else 1.0)
    if not np.isfinite(global_scale) or global_scale <= 0:
        raise ValueError('Invalid NVFP4 global scale')
    blocks = values.reshape(values.shape[0], -1, 16)
    scales = np.max(np.abs(blocks), axis=-1) / (6 * global_scale)
    scales[scales == 0] = 1.0
    scale_codes = fp8_encode(scales.clip(2.0**-9, 448))
    decoded_scales = fp8_decode(scale_codes).astype(np.float32)
    if np.any(decoded_scales <= 0):
        raise ValueError('NVFP4 block scale underflow; calibration/alternative policy required')
    normalized = blocks / (decoded_scales * global_scale)[..., None]
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    codes = nearest(np.abs(normalized), levels)
    codes |= (normalized < 0).astype(np.uint8) << 3
    codes = codes.reshape(values.shape)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    restored = (levels[codes & 7] * np.where(codes & 8, -1, 1)).reshape(blocks.shape)
    restored = (restored * (decoded_scales * global_scale)[..., None]).reshape(values.shape)
    return packed, scale_codes, np.array(global_scale, dtype='<f4'), restored
