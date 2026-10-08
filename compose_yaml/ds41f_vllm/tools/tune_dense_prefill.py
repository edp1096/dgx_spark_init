"""Tune original FlashInfer dense GEMM shapes without resident MoE weights.

Writes a separate candidate cache. Does not replace a serving cache or alter
weights. Both physical ranks must run together so tactic timings are averaged.
"""
import argparse
import ast
from datetime import timedelta
import gc
import json
from pathlib import Path
import time
import torch
import torch.distributed as dist
from flashinfer import mm_mxfp8
from flashinfer.autotuner import AutoTuner, autotune, set_autotune_process_group

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--rank', type=int, choices=(0, 1), required=True)
p.add_argument('--input-cache', type=Path, required=True)
p.add_argument('--output-cache', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--master', default='tcp://10.200.0.1:29613')
a = p.parse_args()
dist.init_process_group('gloo', init_method=a.master, rank=a.rank, world_size=2,
                        timeout=timedelta(minutes=10))
torch.cuda.set_device(0)
torch.manual_seed(20260913+a.rank)
payload = [a.input_cache.read_text() if a.rank == 0 else None]
dist.broadcast_object_list(payload, src=0)
cached = json.loads(payload[0])
queries = [ast.literal_eval(k) for k in cached if k.startswith("('mxfp8_gemm'")]
shapes = sorted({q[2][1] for q in queries})
assert shapes == [(1152,5120),(1280,4096),(1280,16384),(4096,5120),
                  (5120,1792),(5120,2304),(6144,25600),(15360,5120)], shapes
tuner = AutoTuner.get()
# Match vLLM's leader-cache broadcast. The worker's saved file can legitimately
# be older and omit head-only shapes; different hit sets would desynchronize
# the collective tactic search.
canonical = a.output.with_suffix('.input.json')
canonical.write_text(payload[0])
tuner.load_configs(str(canonical))
set_autotune_process_group(dist.group.WORLD)
rows = []


def timing(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000/20


try:
    with torch.inference_mode():
        for k,n in shapes:
            m = 8192
            x = (torch.randn(m,k,device='cuda')*.125).to(torch.float8_e4m3fn)
            w = (torch.randn(n,k,device='cuda')*.125).to(torch.float8_e4m3fn).t()
            blocks = ((k//32+3)//4)*4
            sx = torch.full((m*blocks,),127,dtype=torch.uint8,device='cuda')
            sw = torch.full((((n+127)//128)*128*blocks,),127,dtype=torch.uint8,device='cuda')
            out = torch.empty(m,n,dtype=torch.bfloat16,device='cuda')
            def run():
                return mm_mxfp8(x,w,sx,sw,out=out,backend='cutlass')
            reference = run().clone()
            before = timing(run)
            dist.barrier()
            with autotune(tune_mode=True):
                run()
            after = timing(run)
            actual = run()
            assert bool(actual.isfinite().all())
            # Both paths retain FP8 inputs, FP32 accumulation and BF16 output.
            # Different legal tactics may differ at BF16 rounding boundaries.
            delta = (actual.float()-reference.float()).norm()/reference.float().norm()
            assert delta.item() < 0.0005, ('tactic numerical mismatch',k,n,delta.item())
            row = {'rank':a.rank,'m':m,'k':k,'n':n,'before_ms':before,'after_ms':after,
                   'relative_l2_difference':delta.item(),'max_abs_difference':
                   (actual-reference).abs().max().item()}
            rows.append(row)
            a.output.write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(row),flush=True)
            if a.rank == 0:
                tuner.save_configs(str(a.output_cache))
            del x,w,sx,sw,out,reference,actual,delta
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
    print('DENSE_PREFILL_TUNE_PASS',flush=True)
finally:
    set_autotune_process_group(None)
    dist.destroy_process_group()
