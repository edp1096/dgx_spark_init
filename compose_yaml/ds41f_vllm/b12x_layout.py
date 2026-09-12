"""Pinned b12x V4.1 MXFP4 preparation contract shared by packer and runtime."""
import torch
from b12x.moe import fused_moe

FORMAT='dsv41-b12x-qmma-789bbb3c-tp2-v1'
NAMES=('w1_fp4','w2_fp4','w1_blockscale','w2_blockscale')

def prepare(w13,w2,s13,s2):
    count=w13.shape[0]
    plan=fused_moe.plan_weights(
        source=fused_moe.PackedSource(format='fp4_e8m0_k32',w13_layout='w31'),
        activation=fused_moe.ActivationSpec(mode='a8',nonlinearity='silu',io_dtype=torch.bfloat16,numerical_recipe='deepseek_v41'),
        geometry=fused_moe.MoEGeometry(num_experts=count,hidden_size=5120,intermediate_size=1152))
    unit=torch.ones(count,device=w13.device)
    return fused_moe.prepare_weights(plan=plan,weights=fused_moe.PackedWeights(w13=w13,w2=w2,
        w13_block_scales=s13,w2_block_scales=s2,w13_global_scales=unit,w2_global_scales=unit))

def from_raw(tensors,device='cuda'):
    def load(key):
        raw,shape=tensors[key]
        return torch.frombuffer(bytearray(raw),dtype=torch.uint8).reshape(shape).to(device)
    return prepare(torch.cat((load('w1.weight'),load('w3.weight')),0).unsqueeze(0),
        load('w2.weight').unsqueeze(0),torch.cat((load('w1.scale'),load('w3.scale')),0).unsqueeze(0),load('w2.scale').unsqueeze(0))

def blank(count,device='cuda'):
    return prepare(torch.zeros(count,2304,2560,dtype=torch.uint8,device=device),
        torch.zeros(count,5120,576,dtype=torch.uint8,device=device),
        torch.full((count,2304,160),127,dtype=torch.uint8,device=device),
        torch.full((count,5120,36),127,dtype=torch.uint8,device=device))

def tensors(prepared):
    return [getattr(prepared._impl,name) for name in NAMES]
