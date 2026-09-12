
# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys,time,os
import torch
import torch.nn.functional as F
from expert_store import ExpertStore
from streaming_moe import dequant
import marlin_stream

torch.manual_seed(4)
store=ExpertStore(sys.argv[1],rank=int(os.environ.get("TEST_RANK","1")),cache_bytes=0)
x=torch.randn(3,5120,device='cuda',dtype=torch.bfloat16)*float(os.environ.get('TEST_INPUT_SCALE','.01'))
ids=torch.tensor([[0,1],[1,383],[0,383]],device='cuda',dtype=torch.int32)
weights=torch.tensor([[.3,.7],[.4,.6],[.5,.5]],device='cuda',dtype=torch.float32)
expected=torch.zeros_like(x,dtype=torch.float32)
for expert in ids.unique().tolist():
    row,col=torch.where(ids==expert)
    t=store.expert(0,expert)
    w={n:dequant(t[n+'.weight'],t[n+'.scale'],x.device) for n in ('w1','w2','w3')}
    gate=F.linear(x[row],w['w1']).float().clamp(max=10)
    up=F.linear(x[row],w['w3']).float().clamp(-10,10)
    y=F.linear((F.silu(gate)*up).to(x.dtype),w['w2']).float()*weights[row,col,None]
    expected.index_add_(0,row,y)
for run in range(2):
    torch.cuda.synchronize();start=time.monotonic()
    actual=marlin_stream.apply(store,0,x,weights,ids,10)
    torch.cuda.synchronize()
    relative=(actual.float()-expected).norm()/expected.norm()
    print('run',run,'seconds',time.monotonic()-start,'relative_l2',relative.item(),marlin_stream.stats(),flush=True)
    assert relative < .02
    assert marlin_stream.stats()['gpu_cache_bytes'] <= float(os.environ.get('DSV41_GPU_CACHE_GIB','48'))*2**30
print('MARLIN_STREAM_PASS')
