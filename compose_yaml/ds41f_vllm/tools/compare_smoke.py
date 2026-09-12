"""Pair completed smoke suites and report changed outputs alongside speed."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('before')
p.add_argument('after')
p.add_argument('--output',required=True)
args=p.parse_args()
before=json.loads(Path(args.before).read_text())
after=json.loads(Path(args.after).read_text())
if not before or len(before)!=len(after):
    raise SystemExit('Suites must contain the same nonzero number of requests')
rows=[]
for old,new in zip(before,after):
    if old['prompt']!=new['prompt']: raise SystemExit('Prompt order differs')
    if old['decode_tokens_per_second'] is None or new['decode_tokens_per_second'] is None:
        raise SystemExit('A request has no measurable streamed decode interval')
    rows.append({'prompt':old['prompt'],'identical_output':old['text']==new['text'],
        'before_tokens':old['usage']['completion_tokens'],
        'after_tokens':new['usage']['completion_tokens'],
        'before_decode_tps':old['decode_tokens_per_second'],
        'after_decode_tps':new['decode_tokens_per_second'],
        'decode_speedup':new['decode_tokens_per_second']/old['decode_tokens_per_second'],
        'before_ttft_s':old['ttft_seconds'],'after_ttft_s':new['ttft_seconds'],
        'before_total_s':old['total_seconds'],'after_total_s':new['total_seconds']})
total_before=sum(r['total_seconds'] for r in before)
total_after=sum(r['total_seconds'] for r in after)
summary={'rows':rows,'all_outputs_identical':all(r['identical_output'] for r in rows),
         'total_before_s':total_before,'total_after_s':total_after,
         'end_to_end_speedup':total_before/total_after}
Path(args.output).write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
