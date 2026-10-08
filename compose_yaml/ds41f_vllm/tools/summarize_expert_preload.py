"""Summarize completed measured trials; never turn I/O misses into speed claims."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
raw=json.loads((root/'results/expert-preload-ab.json').read_text())
rows=[r for r in raw['runs'] if not r['warmup']]
summary={'complete_trials':raw['trials'],'profiles':{},'groups':[]}
for count in sorted({r['count'] for r in rows}):
    first=[r for r in rows if r['count']==count and r['repeat']==0]
    if not first:continue
    summary['profiles'][str(count)]={'median_seed_wall_seconds':statistics.median(r['seed_wall_seconds'] for r in first),
       'cache_bytes_per_rank':first[0]['seed'][0]['gpu_cache_bytes'],
       'loaded_bytes_per_rank':first[0]['seed'][0]['read_bytes']}
for fixture in sorted({r['fixture'] for r in rows}):
    for repeat in (0,1):
        group={'fixture':fixture,'repeat':repeat,'modes':{}}
        for count in sorted({r['count'] for r in rows}):
            runs=[r for r in rows if (r['fixture'],r['repeat'],r['count'])==(fixture,repeat,count)]
            if not runs:continue
            output=sum(r['usage']['completion_tokens']-1 for r in runs)
            generated=sum(r['metrics']['generation_time_ms']/1000 for r in runs)
            mode={'n':len(runs),'prompt_tokens':runs[0]['usage']['prompt_tokens'],
                  'cached_tokens':[r['usage']['prompt_tokens_details']['cached_tokens'] for r in runs],
                  'median_ttft_seconds':statistics.median(r['ttft_seconds'] for r in runs),
                  'ttft_range':[min(r['ttft_seconds'] for r in runs),max(r['ttft_seconds'] for r in runs)],
                  'median_elapsed_seconds':statistics.median(r['elapsed_seconds'] for r in runs),
                  'total_generation_tokens':output,'aggregate_tg':output/generated,
                  'all_exact':all(r['exact_output'] for r in runs),
                  'mean_misses_rank0':statistics.mean(r['delta_stats'][0]['slot_misses'] for r in runs)}
            group['modes'][str(count)]=mode
        if '0' in group['modes']:
            base=group['modes']['0']
            for mode in group['modes'].values():
                mode['ttft_reduction_percent']=100*(1-mode['median_ttft_seconds']/base['median_ttft_seconds'])
                mode['tg_change_percent']=100*(mode['aggregate_tg']/base['aggregate_tg']-1)
        summary['groups'].append(group)
(root/'results/expert-preload-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for group in summary['groups']:
 print(group['fixture'],group['repeat'],{k:{key:round(v[key],3) for key in ['median_ttft_seconds','aggregate_tg','ttft_reduction_percent','tg_change_percent']} for k,v in group['modes'].items()})
