#!/usr/bin/env python3
"""Summarize the paired smoke run without claiming statistical significance."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent/'docs'
summary={'scope':{'tp':1,'context':65536,'worker':'192.168.100.60','mem_fraction_static':.79,'engine':'dgx-sglang-qwen38-fn:sm121-vocab1','speed_trials':3,'input_tokens_per_speed_trial':41,'output_tokens_per_speed_trial':256,'temperature':0,'seed':42,'thinking':False},'models':{},'limitations':['Small smoke suite, not a comprehensive quality or uncensoring benchmark.','Only TP1/64K startup tested; long-input retrieval used 6037 prompt tokens, not a full 64K or 1M prompt.','Speed figures are client stream estimates; output text can differ across models. Three trials do not establish statistical significance.']}
for label in ('radix','candidate'):
 results=json.loads((root/f'{label}-runtime.json').read_text())['results'];assert len(results)==14
 assert not any('error' in row for row in results)
 speed=[r for r in results if r['name'].startswith('speed_')]
 assert all(r['usage']['completion_tokens']==256 and r['finish_reason']=='length' for r in speed)
 memory=[json.loads(s) for s in Path(f'/tmp/huihui-{label}-memory.jsonl').read_text().splitlines()]
 summary['models'][label]={'exact_checks_passed':sum(r.get('passed') is True for r in results),'exact_checks_total':sum('passed' in r for r in results),'tg_trials':[r['stream_tg_estimate'] for r in speed],'tg_median':statistics.median(r['stream_tg_estimate'] for r in speed),'ttft_trials':[r['ttft_s'] for r in speed],'ttft_median':statistics.median(r['ttft_s'] for r in speed),'min_available_gib':min(r['available_bytes'] for r in memory if 'available_bytes' in r)/2**30,'guard_tripped':any(r.get('event')=='guard_trip' for r in memory),'oom':any(r.get('oom') for r in memory),'boot_ids':sorted({r['boot_id'] for r in memory}),'needle':next({'prompt_tokens':r['usage']['prompt_tokens'],'ttft_s':r['ttft_s'],'passed':r['passed']} for r in results if r['name']=='long_needle')}
a=summary['models']['radix'];b=summary['models']['candidate'];summary['tg_delta_percent']=(b['tg_median']/a['tg_median']-1)*100
(root/'runtime-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
