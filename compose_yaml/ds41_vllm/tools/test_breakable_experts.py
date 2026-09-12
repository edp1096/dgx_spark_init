"""Captured GPU routing/prefix -> real SSD expert update -> captured suffix."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import os,sys,torch
from contextlib import nullcontext
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import CUDAGraphMode,VllmConfig
from vllm.forward_context import set_forward_context
from expert_store import ExpertStore
import streaming_moe,b12x_slots

torch.manual_seed(57)
streaming_moe._store=ExpertStore(sys.argv[1],rank=0)
x=torch.randn(2,5120,device='cuda').bfloat16()*.1
ids=torch.tensor([[0,1],[2,3]],device='cuda',dtype=torch.int32)
weights=torch.tensor([[.3,.7],[.6,.4]],device='cuda')
full_attention=os.environ.get('DSV41_ATTENTION_GRAPHS')=='1'
config=VllmConfig() if full_attention else None
def forward(n=2):
    activation=x[:n]*2
    routes=ids[:n]+0
    output=torch.empty_like(activation,dtype=torch.float32)
    context=(set_forward_context(None,config,num_tokens=n,
             cudagraph_runtime_mode=CUDAGraphMode.FULL) if full_attention else nullcontext())
    with context:
        streaming_moe.streamed_experts_with_output(activation,weights[:n],routes,output,0,10)
    return output+activation.float()
forward();torch.cuda.synchronize()
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    forward();torch.cuda.synchronize()
    pool=torch.cuda.graph_pool_handle()
    capture=BreakableCUDAGraphCapture(pool=pool)
    with capture: result=forward()
    forward(1);torch.cuda.synchronize()
    small=BreakableCUDAGraphCapture(pool=pool)
    with small: small_result=forward(1)
torch.cuda.current_stream().wait_stream(stream)
assert capture.num_eager_breaks==1 and capture.num_graphs==2
for experts in ([[4,5],[6,7]],[[0,383],[1,382]],[[0,1],[2,3]]):
    ids.copy_(torch.tensor(experts,dtype=torch.int32,device='cuda'))
    x.mul_(.5)
    capture.replay()
    actual=result.clone()
    expected=forward()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    small.replay()
    small_actual=small_result.clone()
    torch.testing.assert_close(small_actual,forward(1),rtol=0,atol=0)
    print('BREAKABLE_REAL_ROUTING_PASS',experts,flush=True)
print(capture,b12x_slots.stats(),flush=True)
