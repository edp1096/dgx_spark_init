"""Summarize matching, uncached prefill trials without mixing in warmups."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json,statistics
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('inputs',nargs='+',type=Path)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
data=sorted([json.loads(path.read_text()) for path in args.inputs],key=lambda d:d['max_batched_tokens'])
baseline={}
summary=[]
fingerprints={}
for suite in data:
    rows=[r for r in suite['runs'] if not r['warmup']]
    cases={r['fixture'] for r in rows}
    settings={'label':suite['label'],'max_batched_tokens':suite['max_batched_tokens'],'kernel_tokens':suite.get('kernel_tokens',512),'shared_buffers':suite.get('shared_buffers',False),
              'routed_pipeline':suite.get('routed_pipeline',False),'image_name':suite.get('image_name'),'cases':{}}
    for name in sorted(cases):
        selected=[r for r in rows if r['fixture']==name]
        assert len(selected)==suite['timed_trials_per_fixture']
        for row in selected:
            assert row['usage']['prompt_tokens_details']['cached_tokens']==0
            assert row['text'].strip().strip('`". *')==row['expected'] and row['finish']=='stop'
        unique={(r['request_sha256'],r['usage']['prompt_tokens']) for r in selected}
        assert len(unique)==1
        key=unique.pop()
        if name in fingerprints:assert fingerprints[name]==key
        fingerprints[name]=key
        tokens=sum(r['usage']['prompt_tokens'] for r in selected)
        prefill=sum(r['metrics']['time_to_first_token_ms']/1000 for r in selected)
        values={'prompt_tokens':key[1],'trials':len(selected),'pp':tokens/prefill,
                'median_ttft_seconds':statistics.median(r['ttft_seconds'] for r in selected),
                'min_pp':min(r['pp'] for r in selected),'max_pp':max(r['pp'] for r in selected),
                'min_host_available_gib':min(v['host_available_gib'] for r in selected for v in r['ranks'].values()),
                'approx_median_expert_read_gib_per_rank':statistics.median(
                    v['last_periodic_counters']['packed_read_bytes']/2**30
                    for r in selected for v in r['ranks'].values() if v['last_periodic_counters'])}
        if name not in baseline:baseline[name]=values
        values['pp_gain_percent']=(values['pp']/baseline[name]['pp']-1)*100
        values['ttft_reduction_percent']=(1-values['median_ttft_seconds']/baseline[name]['median_ttft_seconds'])*100
        settings['cases'][name]=values
    summary.append(settings)
args.output.write_text(json.dumps({'conditions':'Identical requests; zero KV hits; reset expert slots; warmups excluded. Read counts are the last periodic log snapshot.','settings':summary},indent=2)+'\n')
print(args.output.read_text())
