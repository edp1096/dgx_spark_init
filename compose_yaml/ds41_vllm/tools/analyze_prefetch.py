"""Measure forecasts against actual cache misses, without future-label leakage."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

def candidates(predicted,resident,width,budget):
    scores=Counter()
    for row in predicted:
        for rank,expert in enumerate(row[:width]):
            if expert not in resident: scores[expert]+=1/(rank+1)
    return [e for e,_ in scores.most_common(budget)]

p=argparse.ArgumentParser()
p.add_argument('trace')
p.add_argument('--output',required=True)
args=p.parse_args()
rows=[json.loads(line) for line in Path(args.trace).read_text().splitlines() if line]
summary=[]
for variant in ('attention_input','residual_input'):
    for width in (6,12,24,32):
        for budget in (1,2,4,8):
            total=Counter();layers=defaultdict(Counter)
            for row in rows:
                if row['tokens']!=6: continue
                resident=set(row['resident']);needed=set(row['needed'])
                misses=needed-resident
                chosen=candidates(row['predicted'][variant],resident,width,budget)
                useful=len(set(chosen)&misses)
                item={'issued':len(chosen),'useful':useful,'misses':len(misses),'steps':1}
                total.update(item);layers[row['layer']].update(item)
            def report(c):
                return dict(c)|{'precision':c['useful']/max(1,c['issued']),
                    'miss_recall':c['useful']/max(1,c['misses'])}
            summary.append({'variant':variant,'width':width,'budget':budget,
                            **report(total),'layers':{k:report(v) for k,v in layers.items()}})
summary.sort(key=lambda r:(r['miss_recall'],-r['issued']),reverse=True)
Path(args.output).write_text(json.dumps(summary,indent=2))
for row in summary:
    print(json.dumps({k:v for k,v in row.items() if k!='layers'}))
