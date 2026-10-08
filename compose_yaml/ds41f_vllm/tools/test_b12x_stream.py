"""Check native b12x V4.1 MoE against an independent K32 FP8 reference."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys,time
import torch
from b12x.moe import fused_moe
from expert_store import ExpertStore
from streaming_moe import dequant

torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(41)
store=ExpertStore(sys.argv[1],rank=1)
tensors=[store.expert(0,e) for e in (0,1,383)]
def load(t,key):
    raw,shape=t[key]
    return torch.frombuffer(bytearray(raw),dtype=torch.uint8).reshape(shape).cuda()
w13=torch.stack([torch.cat((load(t,'w1.weight'),load(t,'w3.weight')),0) for t in tensors])
s13=torch.stack([torch.cat((load(t,'w1.scale'),load(t,'w3.scale')),0) for t in tensors])
w2=torch.stack([load(t,'w2.weight') for t in tensors])
s2=torch.stack([load(t,'w2.scale') for t in tensors])
plan=fused_moe.plan_weights(source=fused_moe.PackedSource(format='fp4_e8m0_k32',w13_layout='w31'),
 activation=fused_moe.ActivationSpec(mode='a8',nonlinearity='silu',io_dtype=torch.bfloat16,numerical_recipe='deepseek_v41'),
 geometry=fused_moe.MoEGeometry(num_experts=3,hidden_size=5120,intermediate_size=1152))
unit=torch.ones(3,device='cuda')
experts=fused_moe.prepare_weights(plan=plan,weights=fused_moe.PackedWeights(w13=w13,w2=w2,
 w13_block_scales=s13,w2_block_scales=s2,w13_global_scales=unit,w2_global_scales=unit))
execution=fused_moe.plan_execution(experts=experts,capacity=fused_moe.ExecutionCapacity(max_tokens=3,top_k=2))
fused_moe.prewarm(execution)
scratch={spec.name:torch.empty(spec.shape,dtype=spec.dtype,device=spec.device) for spec in execution.scratch_specs()}
x=torch.randn(3,5120,device='cuda').bfloat16()*.1
ids=torch.tensor([[0,1],[1,2],[0,2]],device='cuda',dtype=torch.int32)
weights=torch.tensor([[.3,.7],[.4,.6],[.5,.5]],device='cuda')
out=torch.empty_like(x,dtype=torch.float32)
binding=fused_moe.bind(execution,scratch=scratch,experts=experts,a=x,topk_ids=ids,topk_weights=weights,output=out,input_scales_static=True)
def quant(x):
    blocks=x.float().reshape(*x.shape[:-1],-1,32)
    scale=torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1,keepdim=True).clamp_min(1e-4)/448)))
    return ((blocks/scale).to(torch.float8_e4m3fn).float()*scale).reshape(x.shape)
def mm(a,b):
    out=torch.zeros((a.shape[0],b.shape[0]),device='cuda')
    for k in range(0,a.shape[1],32): out.add_(a[:,k:k+32]@b[:,k:k+32].T)
    return out
expected=torch.zeros_like(out)
for e,t in enumerate(tensors):
    row,col=torch.where(ids==e)
    matrices={n:dequant(t[n+'.weight'],t[n+'.scale'],x.device).float() for n in ('w1','w2','w3')}
    gate=mm(quant(x[row]),matrices['w1']).bfloat16().float().clamp(max=10)
    up=mm(quant(x[row]),matrices['w3']).bfloat16().float().clamp(-10,10)
    mid=(torch.nn.functional.silu(gate)*up*weights[row,col,None]).bfloat16()
    expected.index_add_(0,row,mm(quant(mid),matrices['w2']).bfloat16().float())
for i in range(2):
    torch.cuda.synchronize();start=time.monotonic()
    fused_moe.run(binding=binding)
    torch.cuda.synchronize()
    rel=(out-expected).norm()/expected.norm()
    print('run',i,'seconds',time.monotonic()-start,'relative_l2',rel.item(),'max_abs',(out-expected).abs().max().item(),flush=True)
    torch.testing.assert_close(out,expected,rtol=.01,atol=.08)
print('B12X_NATIVE_PASS')
