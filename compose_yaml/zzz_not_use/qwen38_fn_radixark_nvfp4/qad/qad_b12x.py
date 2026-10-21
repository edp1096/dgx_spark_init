"""SGLang adapters for pinned Apache-2.0 B12X kernels (TP1 experiment)."""
import torch
from torch import nn
from torch.nn import functional as F


def prepare_mxfp8_linear(layer):
    from b12x.gemm.blockscaled import pack_weight
    layer._qad_mxfp8 = pack_weight(layer.weight.detach(), layer.weight_scale_inv.detach(), recipe='mxfp8')


def apply_mxfp8_linear(layer, x, bias=None):
    from b12x.gemm.blockscaled import mm
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        raise ValueError('QAD MXFP8 expects a BF16 activation tensor')
    shape=(*x.shape[:-1],layer._qad_mxfp8.out_features)
    if not x.numel():
        return torch.empty(shape,device=x.device,dtype=x.dtype)
    # Preserve the checkpoint's dynamic MXFP8 activation policy explicitly.
    out=mm(x.reshape(-1,x.shape[-1]).contiguous(),layer._qad_mxfp8,bias=bias,mode='quantized')
    return out.reshape(shape)


def prepare_a16_linear(layer):
    from b12x._lib.intrinsics import swizzle_block_scale
    scales = layer.weight_scale_2.detach().float()
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError('Invalid W4A16 global scale')
    if torch.unique(scales).numel() != 1:
        raise ValueError('W4A16 fused projections need distinct global-scale handling')
    layer._qad_input_k = layer.input_size_per_partition
    padded_k = (layer._qad_input_k + 31) // 32 * 32
    weight = layer.weight.detach()
    block_scales = layer.weight_scale.detach()
    if padded_k != layer._qad_input_k:
        weight = F.pad(weight, (0, (padded_k-layer._qad_input_k)//2))
        # CPU/GPU float8 pad support varies; preserve scale bytes directly.
        block_scales = F.pad(block_scales.view(torch.uint8),
                             (0, (padded_k-layer._qad_input_k)//16)).view(torch.float8_e4m3fn)
    layer.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
    layer.weight_scale = nn.Parameter(swizzle_block_scale(block_scales).contiguous(), requires_grad=False)
    layer.weight_scale_2 = nn.Parameter(scales[:1].contiguous(), requires_grad=False)
    if hasattr(layer, 'input_scale'):
        del layer.input_scale


def apply_a16_linear(layer, x, bias=None):
    from b12x.gemm.blockscaled import w4a16
    if x.dtype != torch.bfloat16:
        raise ValueError('This W4A16 adapter is qualified only for BF16 activations')
    padded_k = layer.weight.shape[1]*2
    source = F.pad(x, (0, padded_k-x.shape[-1])) if x.shape[-1] != padded_k else x.contiguous()
    out = w4a16(source, layer.weight, layer.weight_scale, layer.weight_scale_2)
    if bias is not None:
        out.add_(bias)
    return out
