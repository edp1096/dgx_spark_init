"""Bounded GPU cache of losslessly repacked MXFP4 expert tensors."""
from collections import OrderedDict
import os
from types import SimpleNamespace
import torch
import step_profile
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.scalar_type import scalar_types

_cache=OrderedDict()
_resident=0
_stats={'gpu_hits':0,'gpu_misses':0}

def packed_expert(store, layer, expert, device):
    global _resident
    key=(str(store.model), store.rank, layer, expert, str(device))
    if key in _cache:
        _cache.move_to_end(key)
        _stats['gpu_hits']+=1
        return _cache[key]
    _stats['gpu_misses']+=1
    tensors=store.expert(layer,expert)
    def load(name):
        raw,shape=tensors[name]
        return torch.frombuffer(bytearray(raw),dtype=torch.uint8).reshape(shape).to(device)
    w13=torch.cat((load('w1.weight'),load('w3.weight')),dim=0).unsqueeze(0)
    s13=torch.cat((load('w1.scale'),load('w3.scale')),dim=0).unsqueeze(0)
    data=prepare_moe_mxfp4_layer_for_marlin(SimpleNamespace(params_dtype=torch.bfloat16),
        w13,load('w2.weight').unsqueeze(0),s13,load('w2.scale').unsqueeze(0),None,None)[:4]
    size=sum(t.numel()*t.element_size() for t in data)
    budget=int(float(os.environ.get('DSV41_GPU_CACHE_GIB','48'))*2**30)
    if size <= budget:
        while _cache and _resident+size>budget:
            _,old=_cache.popitem(last=False)
            _resident-=sum(t.numel()*t.element_size() for t in old)
        _cache[key]=data
        _resident+=size
    return data


def apply(store,layer,x,weights,ids,limit):
    if ids.numel() == 0:
        return torch.zeros_like(x)
    with step_profile.phase("lookup_repack"):
        unique,inverse=torch.unique(ids,sorted=True,return_inverse=True)
        selected=[packed_expert(store,layer,e,x.device) for e in unique.tolist()]
    with step_profile.phase("stack_weights"):
        w13,w2,s13,s2=(torch.cat([item[i] for item in selected],dim=0) for i in range(4))
    def activation(kind,out,inp,**kwargs):
        gate,up=inp.float().chunk(2,dim=-1)
        if limit is not None and limit>0:
            gate=gate.clamp(max=limit)
            up=up.clamp(-limit,limit)
        out.copy_((torch.nn.functional.silu(gate)*up).to(out.dtype))
    with step_profile.phase("expert_gemm"):
        result=fused_marlin_moe(x.contiguous(),w13,w2,None,None,s13,s2,
            weights.float(),inverse.to(torch.int32),scalar_types.float4_e2m1f.id,
            activation=MoEActivation.SILU,activation_func=activation)
    return result


def stats():
    return _stats | {'gpu_cache_bytes':_resident}
