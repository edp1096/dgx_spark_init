"""CPU-only ModelOpt conversion, without GGUF decoding or layout permutations."""
import torch
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.quantization.qtensor.mxfp8_tensor import MXFP8QTensor

DTYPES = {'U8': torch.uint8, 'F8_E4M3': torch.float8_e4m3fn,
          'F32': torch.float32, 'BF16': torch.bfloat16, 'F16': torch.float16}


def read(checkpoint, name):
    spec = checkpoint.tensors[name]
    data = bytearray(b''.join(checkpoint.blocks(name)))
    return torch.frombuffer(data, dtype=DTYPES[spec['dtype']]).reshape(spec['shape'])


def blob(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def decode(checkpoint, prefix):
    weight = read(checkpoint, prefix + '.weight')
    scale = read(checkpoint, prefix + '.weight_scale')
    if weight.dtype == torch.float8_e4m3fn:
        if scale.dtype != torch.uint8 or list(scale.shape) != [weight.shape[0], weight.shape[1]//32]:
            raise ValueError('Unexpected MXFP8 geometry')
        result = (weight.float().reshape(weight.shape[0], -1, 32) * torch.exp2(scale.float()-127)[...,None]).reshape(weight.shape)
        return result, 'mxfp8'
    if weight.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
        raise ValueError('Unexpected NVFP4 representation')
    global_scale = read(checkpoint, prefix + '.weight_scale_2')
    if global_scale.numel() != 1 or not torch.isfinite(global_scale).all() or global_scale.item() <= 0:
        raise ValueError('Invalid NVFP4 global scale')
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(weight.shape[0], -1).long()
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    result = (levels[codes].reshape(weight.shape[0], -1, 16) * (scale.float()*global_scale)[...,None]).reshape(weight.shape[0], -1)
    return result, 'nvfp4'


def encode(target, kind, fixed_scales=None):
    if not torch.isfinite(target).all(): raise ValueError('Nonfinite conversion target')
    rounded = target.to(torch.bfloat16)
    if kind == 'mxfp8':
        q, s = MXFP8QTensor.quantize(rounded)
        parts = {'weight': q._quantized_data, 'weight_scale': s}
        restored = q.dequantize(dtype=torch.float32, scale=s)
        choice = 'fresh'
    elif kind == 'nvfp4':
        q, s, g = NVFP4QTensor.quantize(rounded, 16)
        parts = {'weight': q._quantized_data, 'weight_scale': s, 'weight_scale_2': g}
        restored = q.dequantize(dtype=torch.float32, scale=s, double_scale=g, block_sizes={-1:16})
        choice = 'fresh'
        if fixed_scales is not None:
            fq, fs, fg = NVFP4QTensor.quantize(rounded, 16,
                weights_scaling_factor=fixed_scales[0], weights_scaling_factor_2=fixed_scales[1])
            fixed = fq.dequantize(dtype=torch.float32, scale=fs, double_scale=fg, block_sizes={-1:16})
            if torch.sum((fixed-target).double().square()) < torch.sum((restored-target).double().square()):
                parts = {'weight': fq._quantized_data, 'weight_scale': fs, 'weight_scale_2': fg}
                restored, choice = fixed, 'original'
    else:
        raise ValueError('Unknown quantization layout')
    if not torch.isfinite(restored).all(): raise ValueError('Nonfinite quantized result')
    return parts, restored, choice
