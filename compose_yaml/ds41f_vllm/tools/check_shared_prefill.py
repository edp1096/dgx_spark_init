"""Check real packed weights against a frozen pre-change expert implementation."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,importlib.util,importlib.machinery,json,os,time
from pathlib import Path
import torch
import b12x_slots
from expert_store import ExpertStore
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model');p.add_argument('--rank',type=int,required=True)
p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
spec=importlib.util.spec_from_loader('frozen_slots',importlib.machinery.SourceFileLoader('frozen_slots',str(a.reference)))
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
store=ExpertStore(a.model,rank=a.rank)
torch.manual_seed(731)
rows=[]
with torch.inference_mode():
    for layer,topk in ((0,6),(1,6),(40,3)):
        old=reference.SlotLayer(store,layer,topk)
        new=b12x_slots.SlotLayer(store,layer,topk)
        b12x_slots._layers[layer]=new
        for n in (6,511,513,1024,1537,2048,3073,4096):
            active=12 if n in (6,1024) else 48
            x=torch.randn(n,5120,device='cuda',dtype=torch.bfloat16)*.1
            weights=torch.rand(n,topk,device='cuda');weights/=weights.sum(-1,keepdim=True)
            starts=torch.randint(active,(n,1),device='cuda',dtype=torch.int32)
            ids=(starts+torch.arange(topk,device='cuda',dtype=torch.int32))%active
            expected=old.run(x,weights,ids).clone()
            for limit in (512,1024,2048):
                os.environ['DSV41_KERNEL_TOKENS']=str(limit)
                os.environ['DSV41_SHARED_BUFFERS']='1'
                new.reset();before=b12x_slots.stats()['packed_read_bytes']
                actual=new.run(x,weights,ids).clone()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
                read=b12x_slots.stats()['packed_read_bytes']-before
                assert read==len(set(ids.reshape(-1).tolist()))*new.meta['record_bytes']
                # Enqueued consumers own the result before a later layer reuses buffers.
                saved=actual.clone()
                other=next((c for k,c in b12x_slots._layers.items() if k!=layer and c.topk==topk),new)
                other.run(x*2,weights,ids)
                torch.testing.assert_close(actual,saved,rtol=0,atol=0)
                # A separate stream gets distinct graph I/O and scratch storage.
                if n==513 and limit==2048:
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        parallel=new.run(x,weights,ids).clone()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.testing.assert_close(parallel,expected,rtol=1e-5,atol=1e-5)
                rows.append({'layer':layer,'topk':topk,'tokens':n,'kernel_tokens':limit,
                             'max_abs_error':(actual-expected).abs().max().item(),'read_bytes':read})
                print(json.dumps(rows[-1]),flush=True)
        del old
    # Isolate execution from disk I/O, using one fixed resident group.
    cache=b12x_slots._layers[0];n=2048;topk=6
    x=torch.randn(n,5120,device='cuda',dtype=torch.bfloat16)*.1
    weights=torch.full((n,topk),1/topk,device='cuda')
    ids=(torch.arange(n,device='cuda',dtype=torch.int32)[:,None]+torch.arange(topk,device='cuda'))%24
    mapped=cache.ensure(ids.flatten().tolist())
    timings=[]
    for limit in (512,1024,2048):
        os.environ['DSV41_KERNEL_TOKENS']=str(limit)
        for _ in range(3): cache.execute(x,weights,mapped,ids.shape)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(20): cache.execute(x,weights,mapped,ids.shape)
        torch.cuda.synchronize()
        timings.append({'kernel_tokens':limit,'execution_ms':(time.perf_counter()-start)*1000/20})
    result={'rank':a.rank,'rows':rows,'timings':timings,'stats':b12x_slots.stats(),
            'reference_sha256':__import__('hashlib').sha256(a.reference.read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(timings),flush=True)
    print('SHARED_PREFILL_VALIDATION_PASS',flush=True)
