"""Tune native V4.1 scheduling on actual registered expert storage."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json,os,statistics,time
from pathlib import Path
import torch
from b12x.moe import fused_moe
from b12x.moe.fused_moe import _impl
import b12x_slots
from expert_store import ExpertStore

p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--rank',type=int,required=True);p.add_argument('--output',required=True)
args=p.parse_args();torch.manual_seed(842)
store=ExpertStore(args.model,rank=args.rank)
results=[]
with torch.inference_mode():
    for layer,n,topk in ((0,1,6),(0,6,6),(40,5,3),(0,512,6)):
        if layer not in b12x_slots._layers:b12x_slots._layers[layer]=b12x_slots.SlotLayer(store,layer,topk)
        cache=b12x_slots._layers[layer]
        assert cache.direct
        active=min(cache.count,32 if n<100 else cache.count)
        cache.ensure(list(range(active)))
        call=cache.execution(n)
        call['x'].normal_(0,.1)
        routes=torch.arange(call['ids'].numel(),device='cuda',dtype=torch.int32).reshape(call['ids'].shape)%active
        call['ids'].copy_(routes)
        call['weights'].uniform_(.01,.5)
        # Baseline scheduling and every candidate use the identical buffers,
        # checkpoint bytes, precision, routing, and ordered FP32 reduction.
        timings={}
        reference=None
        for clusters in (48,8,16,24,32):
            os.environ['B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS']=str(clusters)
            _impl._MAC_CACHE.clear()
            for _ in range(3):fused_moe.run(binding=call['binding'])
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fused_moe.run(binding=call['binding'])
            graph.replay();torch.cuda.synchronize()
            if reference is None:reference=call['out'].clone()
            else:torch.testing.assert_close(call['out'],reference,rtol=0,atol=0)
            timings[clusters]={'graph':graph,'mac_cache':{str(k):v for k,v in _impl._MAC_CACHE.items()},'samples_ms':[]}
        # Interleave forward/reverse orders after all compilation/capture.
        for order in ((48,8,16,24,32),(32,24,16,8,48)):
            for clusters in order:
                g=timings[clusters]['graph'];samples=timings[clusters]['samples_ms']
                for _ in range(20 if n<100 else 8):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();g.replay();end.record();end.synchronize()
                    samples.append(start.elapsed_time(end))
                torch.testing.assert_close(call['out'],reference,rtol=0,atol=0)
        row={'rank':args.rank,'layer':layer,'tokens':n,'topk':topk,'slots':cache.count,'exact':True,
             'candidates':{str(k):{'median_ms':statistics.median(v['samples_ms']),'samples_ms':v['samples_ms'],'mac_cache':v['mac_cache']} for k,v in timings.items()}}
        results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='candidates'}|{'median_ms':{k:v['median_ms'] for k,v in row['candidates'].items()}}),flush=True)
        Path(args.output).write_text(json.dumps({'b12x_revision':'789bbb3c846565c41f3404af3e0d7c9ce8702f7f','torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'rows':results},indent=2))
        del timings,graph,g
print('MOE_CLUSTER_SWEEP_PASS',flush=True)
