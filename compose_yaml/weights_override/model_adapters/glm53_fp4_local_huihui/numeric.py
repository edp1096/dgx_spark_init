"""Bounded CPU conversion for GLM's MXFP8 and 128x128 FP8 projection weights."""
import math
import re

import numpy as np
import torch


DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "F32": torch.float32}
NAME = re.compile(r"blk\.(\d+)\.(attn_output|ffn_down_shexp)\.weight")


def mapped(row):
    match = NAME.fullmatch(row["name"])
    if not match or not 15 <= int(match[1]) <= 36 or row["type"] != 8:
        raise ValueError("Unsupported Huihui projection: " + row["name"])
    layer, suffix = int(match[1]), match[2]
    columns = 2048 if suffix == "ffn_down_shexp" else (16384 if layer % 4 == 3 else 8192)
    if row["shape"] != [columns, 4096] or row["bytes"] != 4096 * columns // 32 * 34:
        raise ValueError("Unexpected GGUF projection dimensions")
    projection = "mlp.shared_experts.down_proj" if suffix == "ffn_down_shexp" else "self_attn.o_proj"
    kind = "mxfp8" if suffix == "attn_output" and layer % 4 != 3 else "fp8_pb_128"
    return f"model.language_model.layers.{layer}.{projection}", [4096, columns], kind


def q8_decode(data, rows, columns):
    if columns % 32 or len(data) != rows * columns // 32 * 34:
        raise ValueError("Truncated or misaligned Q8_0 tensor")
    blocks = np.frombuffer(data, dtype=np.dtype([("d", "<f2"), ("q", "i1", (32,))]))
    result = blocks["d"].astype(np.float32)[:, None] * blocks["q"].astype(np.float32)
    result = torch.from_numpy(result.reshape(rows, columns).copy())
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite GGUF projection")
    return result


def dequantize(weight, scale):
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise ValueError("Expected a two-dimensional E4M3 matrix")
    rows, columns = weight.shape
    if scale.dtype == torch.uint8:
        if columns % 32 or list(scale.shape) != [rows, columns // 32] or (scale == 255).any():
            raise ValueError("Unexpected UE8M0 scale geometry or reserved scale value")
        factors = torch.exp2(scale.float() - 127)
        result = (weight.float().reshape(rows, -1, 32) * factors[..., None]).reshape(rows, columns)
        kind = "mxfp8"
    elif scale.dtype == torch.float32:
        if rows % 128 or columns % 128 or list(scale.shape) != [rows // 128, 1, columns // 128, 1]:
            raise ValueError("Unexpected 128x128 FP8 scale geometry")
        if not torch.isfinite(scale).all() or (scale < 0).any():
            raise ValueError("Invalid FP8 block scales")
        result = (weight.float().reshape(rows // 128, 128, columns // 128, 128) * scale).reshape(rows, columns)
        kind = "fp8_pb_128"
    else:
        raise ValueError("Unsupported GLM projection scale dtype")
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite reconstructed source projection")
    return result, kind


def _errors(value, target, kind):
    error = (value.double() - target.double()).square()
    rows, columns = value.shape
    if kind == "mxfp8":
        return error.reshape(rows, -1, 32).sum(dim=2)
    return error.reshape(rows // 128, 128, columns // 128, 128).sum(dim=(1, 3), keepdim=True)


def _weight_mask(mask, shape, kind):
    rows, columns = shape
    if kind == "mxfp8":
        return mask[..., None].expand(rows, columns // 32, 32).reshape(shape)
    return mask.expand(rows // 128, 128, columns // 128, 128).reshape(shape)


def candidates(target, weight, scale):
    """Return unchanged, original-scale, and fresh-scale candidates in that order."""
    base, kind = dequantize(weight, scale)
    rounded = target.to(torch.bfloat16).float()
    if not torch.isfinite(rounded).all():
        raise ValueError("BF16 target overflow")
    if kind == "mxfp8":
        from modelopt.torch.quantization.qtensor.mxfp8_tensor import MXFP8QTensor

        q, fresh_scale = MXFP8QTensor.quantize(rounded)
        fresh = q._quantized_data
        factors = torch.exp2(scale.float() - 127)
        fixed = (rounded.reshape(weight.shape[0], -1, 32) / factors[..., None])
        fixed = fixed.clamp(-448, 448).to(torch.float8_e4m3fn).reshape(weight.shape)
    else:
        from modelopt.torch.quantization.qtensor.fp8_tensor import FP8QTensor

        rows, columns = weight.shape
        blocks = rounded.reshape(rows // 128, 128, columns // 128, 128)
        fresh_scale = blocks.abs().amax(dim=(1, 3), keepdim=True) / 448.0
        fresh_scale = torch.where(fresh_scale > 0, fresh_scale, torch.ones_like(fresh_scale))
        q, _ = FP8QTensor.quantize(
            rounded, scales=fresh_scale.reshape(rows // 128, columns // 128),
            block_sizes={0: 128, 1: 128},
        )
        fresh = q._quantized_data
        divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
        fixed = (blocks / divisor).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(weight.shape)
    values = [(weight, scale, base)]
    for q, s in [(fixed, scale), (fresh, fresh_scale)]:
        reconstructed, resolved = dequantize(q, s)
        if resolved != kind:
            raise ValueError("Quantizer changed the projection format")
        values.append((q, s, reconstructed))
    return kind, values


def requantize(target, weight, scale):
    """Choose the nearest candidate per block, keeping source bytes on ties."""
    kind, values = candidates(target, weight, scale)
    selected_weight = weight.view(torch.uint8).clone()
    selected_scale = scale.clone()
    error = _errors(values[0][2], target, kind)
    choice = torch.zeros_like(error, dtype=torch.uint8)
    for number, (q, s, reconstructed) in enumerate(values[1:], 1):
        new_error = _errors(reconstructed, target, kind)
        select = new_error < error
        selected_weight = torch.where(_weight_mask(select, weight.shape, kind),
                                      q.view(torch.uint8), selected_weight)
        selected_scale = torch.where(select, s, selected_scale)
        error = torch.where(select, new_error, error)
        choice = torch.where(select, torch.full_like(choice, number), choice)
    selected_weight = selected_weight.view(torch.float8_e4m3fn)
    restored, resolved = dequantize(selected_weight, selected_scale)
    if kind != resolved:
        raise ValueError("Selected result changed precision")
    counts = {name: int((choice == i).sum()) for i, name in enumerate(
        ("unchanged_blocks", "original_scale_blocks", "fresh_scale_blocks"))}
    return selected_weight, selected_scale, restored, counts


def norm2(value):
    return float(value.double().square().sum())


def transform(original, donor, weight, scale):
    base, kind = dequantize(weight, scale)
    if original.shape != base.shape or donor.shape != base.shape:
        raise ValueError("GGUF and checkpoint projection shapes differ")
    delta = donor - original
    target = base + delta
    if not torch.isfinite(target).all():
        raise ValueError("Nonfinite ablation target")
    q, s, restored, counts = requantize(target, weight, scale)
    effect = restored - base
    stats = {
        "base_norm2": norm2(base), "original_norm2": norm2(original),
        "source_alignment_error2": norm2(base - original),
        "delta_norm2": norm2(delta), "target_norm2": norm2(target),
        "error_norm2": norm2(target - restored), "effect_norm2": norm2(effect),
        "delta_dot_effect": float((delta.double() * effect.double()).sum()), **counts,
    }
    return q, s, kind, stats


def summarize(stats):
    ratio = lambda numerator, denominator: math.sqrt(stats[numerator] / max(stats[denominator], 1e-300))
    return {
        **stats,
        "source_alignment_relative_l2": ratio("source_alignment_error2", "original_norm2"),
        "delta_relative_l2": ratio("delta_norm2", "base_norm2"),
        "target_relative_error": ratio("error_norm2", "target_norm2"),
        "error_relative_to_delta": ratio("error_norm2", "delta_norm2"),
        "effect_relative_to_delta": ratio("effect_norm2", "delta_norm2"),
        "delta_effective_cosine": stats["delta_dot_effect"] / max(
            math.sqrt(stats["delta_norm2"] * stats["effect_norm2"]), 1e-300),
    }
