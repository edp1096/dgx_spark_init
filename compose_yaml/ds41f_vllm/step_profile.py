"""Opt-in synchronized phase timings, including speculative verification."""
from contextlib import contextmanager
import json
import os
import time
import torch

active=False
parts={}
last_end=None
step_start=None
step_tokens=None

def begin(layer, tokens):
    global active,parts,last_end,step_start,step_tokens
    enabled=os.path.exists('/opt/ds41/profile.enabled') and tokens <= 16
    if layer == 0:
        active=enabled
        parts={}
        last_end=None
        if active:
            step_tokens=tokens
            torch.cuda.synchronize()
            step_start=time.perf_counter()
    if active:
        torch.cuda.synchronize()
        if last_end is not None:
            parts['between_moe']=parts.get('between_moe',0)+time.perf_counter()-last_end

@contextmanager
def phase(name):
    if not active:
        yield
        return
    torch.cuda.synchronize()
    start=time.perf_counter()
    yield
    torch.cuda.synchronize()
    parts[name]=parts.get(name,0)+time.perf_counter()-start

def end(layer):
    global last_end,active
    if not active:
        return
    torch.cuda.synchronize()
    last_end=time.perf_counter()
    if layer == 39:
        from vllm.distributed import get_tensor_model_parallel_rank
        print('STEP_PROFILE '+json.dumps({'rank':get_tensor_model_parallel_rank(),'tokens':step_tokens,
              'total_ms':(last_end-step_start)*1000, **{k+'_ms':v*1000 for k,v in parts.items()}}),flush=True)
        active=False
