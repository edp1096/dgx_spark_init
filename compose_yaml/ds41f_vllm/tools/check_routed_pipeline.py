"""Validate actual-route staging against frozen resident-first execution."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,importlib.util,importlib.machinery,json,os
from pathlib import Path
import torch
import b12x_slots,routed_pipeline
from expert_store import ExpertStore
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model');p.add_argument('--rank',type=int,required=True)
p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
spec=importlib.util.spec_from_loader('reference',importlib.machinery.SourceFileLoader('reference',str(a.reference)))
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
store=ExpertStore(a.model,rank=a.rank);torch.manual_seed(912)
rows=[]
with torch.inference_mode():
    old=reference.SlotLayer(store,0,6)
    new=b12x_slots.SlotLayer(store,0,6);b12x_slots._layers[0]=new
    for n,active in ((513,384),(1537,256),(2048,384),(3073,384)):
        x=torch.randn(n,5120,device='cuda',dtype=torch.bfloat16)*.1
        weights=torch.rand(n,6,device='cuda');weights/=weights.sum(-1,keepdim=True)
        ids=(torch.arange(n,device='cuda',dtype=torch.int32)[:,None]+torch.arange(6,device='cuda'))%active
        expected=old.run(x,weights,ids).clone()
        new.reset()
        for phase in ('cold','warm'):
            before=b12x_slots.stats();missing=len(set(ids.flatten().tolist())-set(new.used))
            actual=new.run(x,weights,ids).clone();torch.cuda.synchronize()
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
            read=b12x_slots.stats()['packed_read_bytes']-before['packed_read_bytes']
            assert read==missing*new.meta['record_bytes'],(read,missing)
            rows.append({'tokens':n,'active':active,'phase':phase,'missing':missing,'read_bytes':read,
                         'max_abs_error':(actual-expected).abs().max().item()})
            print(json.dumps(rows[-1]),flush=True)
    # Failure after the staging reader is submitted must drain writers.
    def fail(): raise RuntimeError('INJECTED_CALLBACK_FAILURE')
    try: new.run(x,weights,ids,on_load_start=fail)
    except RuntimeError as e: assert str(e)=='INJECTED_CALLBACK_FAILURE'
    else: raise AssertionError('Callback failure did not fire')
    new.reset();actual=new.run(x,weights,ids).clone()
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
    del old
    # A different stream receives a distinct bank and waits for shared slot use.
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): parallel=new.run(x,weights,ids).clone()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(parallel,expected,rtol=1e-5,atol=1e-5)
    assert routed_pipeline.stats()['staging_banks']==2
    # Another layer on the original stream reuses its bank only after consumers.
    other=b12x_slots.SlotLayer(store,1,6);b12x_slots._layers[1]=other
    other.run(x,weights,ids)
    actual=new.run(x,weights,ids).clone()
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
    assert routed_pipeline.stats()['staging_banks']==2
    result={'rank':a.rank,'rows':rows,'callback_failure_drained':True,'cross_stream_and_layer':True,'stats':b12x_slots.stats()}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('ROUTED_PIPELINE_VALIDATION_PASS',flush=True)
