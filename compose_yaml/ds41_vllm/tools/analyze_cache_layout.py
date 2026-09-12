"""Replay actual routing and fit a fixed total slot budget on a training trace."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json
from collections import Counter,OrderedDict,defaultdict
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--output',required=True)
args=p.parse_args();rows=[json.loads(line) for line in Path(args.trace).read_text().splitlines()]

def replay(records,capacity,protected_fraction=0):
    used=OrderedDict();protected=OrderedDict();misses=0;hits=0
    def ensure(needed):
        nonlocal misses,hits
        requested=set(needed);missing=[]
        for expert in dict.fromkeys(needed):
            if expert in used:
                used.move_to_end(expert);hits+=1
                if protected_fraction:
                    protected[expert]=None;protected.move_to_end(expert)
                    while len(protected)>int(capacity*protected_fraction):protected.popitem(last=False)
            else:missing.append(expert)
        # Reserve all destinations before publishing any new entry, matching
        # SlotLayer.ensure's protected demand set and batched read completion.
        occupied=len(used)
        free=capacity-occupied
        for _ in missing:
            if free:free-=1
            else:
                victim=next((e for e in used if e not in requested and e not in protected),None)
                if victim is None:victim=next(e for e in used if e not in requested)
                used.pop(victim);protected.pop(victim,None)
        for expert in missing:used[expert]=None
        misses+=len(missing)
    for row in records:
        needed=row['needed'];unique=set(needed)
        if len(unique)<=capacity:ensure(needed)
        else:
            topk=len(needed)//row['tokens']
            frequency=Counter(needed[-64*topk:])
            ordered=sorted(unique,key=lambda e:(frequency[e],e))
            for start in range(0,len(ordered),capacity):ensure(ordered[start:start+capacity])
    return {'misses':misses,'hits':hits}

layers=defaultdict(list)
for row in rows:layers[row['layer']].append(row)
assert all(layer in layers for layer in range(40))
capacities=list(range(64,385,16));budget=40*224
curves={layer:{size:replay(layers[layer],size)['misses'] for size in capacities} for layer in range(40)}
# Exact bounded knapsack: minimize measured training misses under the same
# 8,960 target slots. No future route is used at serving time.
states={0:(0,[])}
for layer in range(40):
    next_states={}
    remaining=39-layer
    for used,(cost,allocation) in states.items():
        for size in capacities:
            total=used+size
            if not total+remaining*64<=budget<=total+remaining*384:continue
            value=cost+curves[layer][size]
            if total not in next_states or value<next_states[total][0]:
                next_states[total]=(value,allocation+[size])
    states=next_states
cost,allocation=states[budget]
baseline=sum(curves[layer][224] for layer in range(40))
slru={fraction:sum(replay(layers[layer],224,fraction)['misses'] for layer in range(40)) for fraction in (.5,.75,.9)}
result={'slru_uniform_training_misses':slru,'schema':1,'source_trace':Path(args.trace).name,'target_slots':allocation,'total_target_slots':sum(allocation),
        'training_baseline_misses':baseline,'training_candidate_misses':cost,
        'training_miss_reduction_percent':100*(1-cost/baseline),'curves':curves,
        'baseline_all_layers':{layer:replay(records,224 if layer<40 else 128) for layer,records in layers.items()}}
Path(args.output).write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('curves','baseline_all_layers')},indent=2))
