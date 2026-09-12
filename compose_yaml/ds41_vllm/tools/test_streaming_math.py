"""Compare streamed TP partial expert GEMMs with a full expert computation."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open
from expert_store import ExpertStore
from streaming_moe import dequant

torch.set_num_threads(2)
torch.manual_seed(17)
model = sys.argv[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a, b = ExpertStore(model,0), ExpertStore(model,1)
x = torch.randn(2,5120,device=device,dtype=torch.bfloat16)*.01
mix = torch.tensor([[.2],[.7]],device=device)

def run(tensors):
    w={name:dequant(tensors[f'{name}.weight'], tensors[f'{name}.scale'],device) for name in ('w1','w2','w3')}
    gate=F.linear(x,w['w1']).float().clamp(max=10)
    up=F.linear(x,w['w3']).float().clamp(-10,10)
    return F.linear((F.silu(gate)*up*mix).to(x.dtype),w['w2']).float()

for layer,expert in [(0,0),(39,383)]:
    tensors={}
    for w in ('w1','w2','w3'):
        for kind in ('weight','scale'):
            name=f'layers.{layer}.ffn.experts.{expert}.{w}.{kind}'
            path,_,_=a.index[name]
            with safe_open(str(path),framework='pt',device='cpu') as f:
                t=f.get_tensor(name).view(torch.uint8).contiguous()
                tensors[f'{w}.{kind}']=(t.numpy().tobytes(),tuple(t.shape))
    expected=run(tensors)
    actual=run(a.expert(layer,expert))+run(b.expert(layer,expert))
    # BF16 per-rank rounding differs from full GEMM; assess normalized error.
    relative=(actual-expected).norm()/expected.norm().clamp_min(1e-10)
    print(layer,expert,'relative_l2',relative.item(),flush=True)
    assert relative < .02
print('PASS: streamed TP expert sum vs full original expert, BF16 rounding tolerance')
