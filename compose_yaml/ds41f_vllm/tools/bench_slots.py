
# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys,time,os,json
import torch
from expert_store import ExpertStore
import b12x_slots
store=ExpertStore(sys.argv[1],rank=0)
x=torch.randn(1,5120,device='cuda',dtype=torch.bfloat16)*.1
ids=torch.arange(6,device='cuda',dtype=torch.int32).reshape(1,6)
weights=torch.full((1,6),1/6,device='cuda')
b12x_slots.apply(store,0,x,weights,ids,10)
torch.cuda.synchronize()
outputs=[]
for graph in ['0','1']:
    os.environ['DSV41_EXPERT_GRAPHS']=graph
    for _ in range(3): b12x_slots.apply(store,0,x,weights,ids,10)
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(50): result=b12x_slots.apply(store,0,x,weights,ids,10)
    torch.cuda.synchronize()
    outputs.append(result.clone())
    print(json.dumps({'graph':graph,'ms_per_layer':(time.perf_counter()-start)*1000/50,**b12x_slots.stats()}),flush=True)
torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
print('GRAPH_EAGER_PARITY_PASS')
