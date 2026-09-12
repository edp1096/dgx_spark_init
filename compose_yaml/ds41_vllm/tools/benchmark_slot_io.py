"""Compare miss transport and warm graph costs on identical expert sequences."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,gc,json,os,time,statistics
from pathlib import Path
import torch
from expert_store import ExpertStore
from b12x_slots import SlotLayer

p=argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--modes',nargs='+',default=['buffered','direct'])
p.add_argument('--output',default='/tmp/slot-io.json')
args=p.parse_args()
store=ExpertStore(args.model,rank=int(os.environ.get('DSV41_TEST_RANK','0')))
torch.manual_seed(41)
x=torch.randn(6,5120,device='cuda').bfloat16()*.1
weights=torch.full((6,6),1/6,device='cuda')
ids=torch.tensor([list(range(6))]*6,device='cuda',dtype=torch.int32)
results=[];reference=None
for mode in args.modes:
    os.environ['DSV41_SLOT_IO']=mode
    cache=SlotLayer(store,0,6)
    cache.run(x,weights,ids)
    torch.cuda.synchronize()
    out=cache.run(x,weights,ids).clone()
    if reference is None: reference=out
    else: torch.testing.assert_close(out,reference,rtol=0,atol=0)
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(100): cache.run(x,weights,ids)
    torch.cuda.synchronize();warm_ms=(time.perf_counter()-start)*10
    timings={}
    for size in (6,24):
        seconds=[]
        for i in range(12):
            wanted=list(range(32+i*24,32+i*24+size))
            torch.cuda.synchronize();start=time.perf_counter()
            cache.ensure(wanted)
            torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
        timings[str(size)]={'median_ms':statistics.median(seconds)*1000,
             'mean_ms':sum(seconds)/len(seconds)*1000,
             'logical_gb_s':size*cache.meta['record_bytes']*len(seconds)/sum(seconds)/1e9,
             'samples_ms':[s*1000 for s in seconds]}
    item={'mode':mode,'warm_graph_ms':warm_ms,'misses':timings}
    results.append(item);print(json.dumps(item),flush=True)
    os.close(cache.fd);del cache
    gc.collect();torch.cuda.empty_cache()
Path(args.output).write_text(json.dumps(results,indent=2))
