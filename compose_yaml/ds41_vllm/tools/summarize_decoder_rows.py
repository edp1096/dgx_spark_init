"""Summarize paired decoder-expert measurements without mixing warmups."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json,statistics
from pathlib import Path
r=Path(__file__).resolve().parents[1]/'results'
data=json.loads((r/'decoder-rows-ab.json').read_text())
summary={'fixtures':{},'timed_trials_per_mode':data['trials']}
for fixture in sorted({x['fixture'] for x in data['runs']}):
    modes={}
    for enabled in (False,True):
        rows=[x for x in data['runs'] if x['fixture']==fixture and x['enabled']==enabled and not x['warmup']]
        assert len(rows)==data['trials'],(fixture,enabled,len(rows))
        assert len({x['request_sha256'] for x in rows})==1
        tokens=sum(x['reply']['usage']['prompt_tokens'] for x in rows)
        modes[str(enabled)]={'input_tokens':rows[0]['reply']['usage']['prompt_tokens'],
                             'weighted_pp':tokens/sum(x['reply']['metrics']['time_to_first_token_ms']/1000 for x in rows),
                             'median_ttft_seconds':statistics.median(x['ttft_seconds'] for x in rows),
                             'ttft_range_seconds':[min(x['ttft_seconds'] for x in rows),max(x['ttft_seconds'] for x in rows)]}
    base,changed=modes['False'],modes['True']
    summary['fixtures'][fixture]={'baseline':base,'selected_experts':changed,
        'pp_gain_percent':100*(changed['weighted_pp']/base['weighted_pp']-1),
        'ttft_reduction_percent':100*(1-changed['median_ttft_seconds']/base['median_ttft_seconds'])}
p=r/'decoder-rows-continuation.json'
if p.exists():
    rows=json.loads(p.read_text());tg={}
    for enabled in (False,True):
        subset=[x for x in rows if x['kind']=='decode' and x['enabled']==enabled]
        if subset:
            generated=sum(x['reply']['usage']['completion_tokens']-1 for x in subset)
            seconds=sum(x['reply']['metrics']['generation_time_ms']/1000 for x in subset)
            tg[str(enabled)]={'weighted_tg':generated/seconds,'cases':len(subset)}
    summary['decode_check']=tg
fixed=r/'decoder-fixed-output.json'
if fixed.exists():
    rows=json.loads(fixed.read_text())
    assert len(rows)==8 and all(x['exact_output'] for x in rows)
    stats={}
    for enabled in (False,True):
        group=[x for x in rows if x['enabled']==enabled]
        count=sum(x['reply']['usage']['completion_tokens']-1 for x in group)
        seconds=sum(x['reply']['metrics']['generation_time_ms']/1000 for x in group)
        stats[str(enabled)]={'cases':len(group),'generated_tokens_after_first':count,
                             'seconds':seconds,'tg':count/seconds}
    assert stats['False']['generated_tokens_after_first']==stats['True']['generated_tokens_after_first']
    stats['change_percent']=100*(stats['True']['tg']/stats['False']['tg']-1)
    summary['fixed_output_tg']=stats
    summary['decision']='default_off: PP gain does not justify measured TG regression'
(r/'decoder-rows-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
