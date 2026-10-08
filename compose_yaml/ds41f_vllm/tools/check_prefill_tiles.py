"""Compare large resident/evicting prefills with the qualified 512-token path."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter
import json
from pathlib import Path
import torch
import b12x_slots
from expert_store import ExpertStore

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('model')
parser.add_argument('--rank',type=int,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
torch.manual_seed(6142)
store=ExpertStore(args.model,rank=args.rank)
rows=[]

def legacy_chunk(cache,x,weights,ids):
    """Frozen pre-change 512-token grouping, independent of the new _run."""
    needed=ids.reshape(-1).tolist()
    unique=set(needed)
    if len(unique)<=cache.count:
        return cache.execute(x,weights,cache.ensure(needed),ids.shape)
    frequency=Counter(needed[-64*cache.topk:])
    ordered=sorted(unique,key=lambda e:(frequency[e],e))
    output=torch.zeros_like(x,dtype=torch.float32)
    for begin in range(0,len(ordered),cache.count):
        group=ordered[begin:begin+cache.count]
        remap=dict(zip(group,cache.ensure(group)))
        output.add_(cache.execute(x,weights,[remap.get(e,-1) for e in needed],ids.shape))
    return output

with torch.inference_mode():
    cache=b12x_slots.SlotLayer(store,0,6)
    b12x_slots._layers[0]=cache
    for n,active in ((512,48),(1024,12),(1537,48),(2048,48)):
        x=torch.randn(n,5120,device='cuda',dtype=torch.bfloat16)*.1
        weights=torch.rand(n,6,device='cuda')
        weights=weights/weights.sum(-1,keepdim=True)
        # No duplicate expert within one token; include a cache-evicting case.
        starts=torch.randint(active,(n,1),device='cuda',dtype=torch.int32)
        ids=(starts+torch.arange(6,device='cuda',dtype=torch.int32))%active
        reference=[]
        for start in range(0,n,512):
            reference.append(legacy_chunk(cache,x[start:start+512],weights[start:start+512],ids[start:start+512]).clone())
        reference=torch.cat(reference)
        cache=b12x_slots._layers[0]
        assert cache.count==24
        cache.reset()
        before=b12x_slots.stats()['packed_read_bytes']
        actual=b12x_slots.apply(store,0,x,weights,ids,24)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-5)
        # Whole-batch grouping must read each active expert only once.
        read_bytes=b12x_slots.stats()['packed_read_bytes']-before
        assert read_bytes==active*cache.meta['record_bytes'],read_bytes
        saved=actual.clone()
        b12x_slots.apply(store,0,x[:6]*2,weights[:6],ids[:6],24)
        torch.testing.assert_close(actual,saved,rtol=0,atol=0)
        # Start with useful experts still in the cache. They must be consumed
        # before any missing group can evict them and force a redundant read.
        missing=len(set(ids.reshape(-1).tolist())-set(cache.used))
        before=b12x_slots.stats()['packed_read_bytes']
        warm=b12x_slots.apply(store,0,x,weights,ids,24)
        torch.cuda.synchronize()
        torch.testing.assert_close(warm,reference,rtol=1e-5,atol=1e-5)
        warm_reads=b12x_slots.stats()['packed_read_bytes']-before
        assert warm_reads==missing*cache.meta['record_bytes'],(warm_reads,missing)
        rows.append({'tokens':n,'active_experts':active,'max_abs_error':(actual-reference).abs().max().item(),
                     'read_bytes':read_bytes,'one_read_per_expert':True,'output_survives_next_replay':True,
                     'warm_missing_experts':missing,'warm_read_bytes':warm_reads,'resident_experts_not_reread':True})
        print(json.dumps(rows[-1]),flush=True)
args.output.write_text(json.dumps({'rank':args.rank,'rows':rows,'scratch_bytes':b12x_slots.stats()['scratch_bytes']},indent=2)+'\n')
print('PREFILL_TILE_VALIDATION_PASS',flush=True)
