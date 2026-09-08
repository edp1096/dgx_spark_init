#!/usr/bin/env python3
"""Summarize completed broad-domain comparisons without inventing usage weights."""
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

from vocab_memory import save


def geomean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def expand(path):
    return {i for start, end in json.loads(path.read_text())['ranges'] for i in range(start, end)}


def coverage(ids, sets):
    n=len(ids)
    outside={mode:sum(i not in vocab for i in ids) for mode,vocab in sets.items()}
    return dict(tokens=n, outside=outside,
                outside_pct={k:100*v/n if n else 0 for k,v in outside.items()},
                recovered_by_128k=sum(i not in sets['ko64k'] and i in sets['ko128k'] for i in ids))


def aggregate(rows):
    tokens=sum(len(r['output_ids']) for r in rows)
    proposed=sum(r['meta_info']['spec_num_proposed_drafts'] for r in rows)
    accepted=sum(r['meta_info']['spec_num_correct_drafts'] for r in rows)
    verifies=sum(r['meta_info']['spec_verify_ct'] for r in rows)
    decode_seconds=sum((len(r['output_ids'])-r['first_chunk_tokens'])/r['decode_tok_s'] for r in rows)
    decode_tokens=sum(len(r['output_ids'])-r['first_chunk_tokens'] for r in rows)
    return dict(requests=len(rows), tokens=tokens,
                elapsed_s=sum(r['elapsed_s'] for r in rows),
                pooled_end_to_end_tok_s=tokens/sum(r['elapsed_s'] for r in rows),
                pooled_decode_tok_s=decode_tokens/decode_seconds,
                mean_request_decode_tok_s=statistics.mean(r['decode_tok_s'] for r in rows),
                median_ttft_s=statistics.median(r['ttft_s'] for r in rows),
                spec_accept_rate=accepted/proposed, spec_accept_length=tokens/verifies,
                truncated=sum(r['truncated'] for r in rows))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
    out=args.directory.resolve()
    assert (out/'complete.json').exists(), 'Run incomplete'
    tasks=json.loads((out/'tasks.json').read_text())['tasks']
    taskhash=hashlib.sha256((out/'tasks.json').read_bytes()).hexdigest()
    close_think=json.loads((out/'token-metadata.json').read_text())['special_token_ids']['</think>']
    maps=out.parent/'2026-09-08-vocab-memory'
    sets={m:expand(maps/f'{m}.json') for m in ['ko64k','ko128k']}
    assert sets['ko64k'] <= sets['ko128k']
    infos={m:json.loads((out/f'{m}-server-info.json').read_text()) for m in sets}
    for key in ['model_path','tokenizer_path','context_length','max_total_tokens',
                'max_mamba_cache_size','speculative_algorithm','speculative_num_steps',
                'speculative_num_draft_tokens','speculative_eagle_topk','mem_fraction_static']:
        assert infos['ko64k'].get(key)==infos['ko128k'].get(key), key
    assert infos['ko64k']['context_length']==65536
    assert infos['ko64k']['max_total_tokens']==65536
    assert infos['ko64k']['max_mamba_cache_size']==18
    allrows={}
    per_mode={}
    for mode in sets:
        data=json.loads((out/f'{mode}-benchmark.json').read_text())
        assert data['tasks_sha256']==taskhash
        rows=data['rows'];allrows[mode]=rows
        assert len(rows)==48 and len({(r['id'],r['repeat']) for r in rows})==48
        assert {r['id'] for r in rows}=={t['id'] for t in tasks}
        for row in rows:
            assert len(row['output_ids'])==row['meta_info']['completion_tokens']
            row['coverage']=coverage(row['output_ids'],sets)
            ids=row['output_ids']
            if row['thinking']:
                if close_think in ids:
                    split=ids.index(close_think)
                    row['reasoning_coverage']=coverage(ids[:split],sets)
                    row['answer_coverage']=coverage(ids[split+1:],sets)
                    row['answer_present']=bool(ids[split+1:])
                else:
                    row['reasoning_coverage']=coverage(ids,sets)
                    row['answer_coverage']=coverage([],sets)
                    row['answer_present']=False
            else:
                row['answer_coverage']=coverage(ids,sets);row['answer_present']=True
        mem=json.loads((out/f'{mode}-memory.json').read_text())
        state=json.loads((out/f'{mode}-container-state.json').read_text())
        assert not mem['aborted'] and not mem['monitor_error'] and not state['OOMKilled']
        telemetry=[json.loads(x) for x in (out/f'{mode}-telemetry.jsonl').read_text().splitlines()]
        assert not any(int(r.get('memory.events',{}).get('oom_kill',0)) for r in telemetry)
        per_mode[mode]=dict(**aggregate(rows),
            coverage=coverage([i for r in rows for i in r['output_ids']],sets),
            answer_coverage=coverage([i for r in rows for i in
                (r['output_ids'][r['output_ids'].index(close_think)+1:] if r['thinking'] and close_think in r['output_ids']
                 else [] if r['thinking'] else r['output_ids'])],sets),
            missing_answers=sum(not r['answer_present'] for r in rows),
            min_available_gib=min(r['MemAvailable'] for r in telemetry)/2**30,
            peak_extra_memory_gib=(mem['baseline']['MemAvailable']-min(r['MemAvailable'] for r in telemetry))/2**30,
            post_idle_extra_memory_gib=mem['phases']['post_idle']['median_extra_memory_gib'],
            max_swap_growth_mib=max(0,mem['baseline']['SwapFree']-min(r['SwapFree'] for r in telemetry))/2**20)
        thermal_path=out/'thermal.jsonl'
        if thermal_path.exists():
            intervals=[(datetime.datetime.fromisoformat(r['started']),
                        datetime.datetime.fromisoformat(r['finished'])) for r in rows]
            thermal_paths=[thermal_path]+list((out/'attempts').glob('*/thermal.jsonl'))
            observations=[json.loads(x) for path in thermal_paths for x in path.read_text().splitlines()]
            observations=[r for r in observations if any(a<=datetime.datetime.fromisoformat(r['time'])<=b
                                                       for a,b in intervals)]
            thermals={}
            for key in ['temperature_c','gpu_util_pct','power_w','sm_clock_mhz']:
                values=[]
                for obs in observations:
                    try:values.append(float(obs.get('values',{}).get(key,'')))
                    except ValueError:pass
                if values:
                    thermals[key]=dict(samples=len(values),min=min(values),max=max(values),median=statistics.median(values))
            per_mode[mode]['thermal_during_requests']=thermals
    per_task=[]
    for task in tasks:
        matching={m:sorted([r for r in rows if r['id']==task['id']],key=lambda r:r['repeat']) for m,rows in allrows.items()}
        ratios=[b['decode_tok_s']/a['decode_tok_s'] for a,b in zip(matching['ko64k'],matching['ko128k'])]
        complete_ratios=[b['decode_tok_s']/a['decode_tok_s'] for a,b in zip(matching['ko64k'],matching['ko128k'])
                         if not a['truncated'] and not b['truncated'] and a['answer_present'] and b['answer_present']]
        per_task.append(dict(id=task['id'],domain=task['domain'],thinking=task['thinking'],
            decode_ratio_128_over_64=geomean(ratios), repeat_decode_ratios=ratios,
            complete_pairs=len(complete_ratios),
            complete_pair_decode_ratio_128_over_64=geomean(complete_ratios) if complete_ratios else None,
            elapsed_ratio_128_over_64=geomean([b['elapsed_s']/a['elapsed_s'] for a,b in zip(matching['ko64k'],matching['ko128k'])]),
            exact_output_match_pairs=sum(a['output_ids']==b['output_ids'] for a,b in zip(matching['ko64k'],matching['ko128k'])),
            modes={m:dict(**aggregate(rows),coverage=coverage([i for r in rows for i in r['output_ids']],sets)) for m,rows in matching.items()}))
    per_domain=[]
    for domain in sorted({t['domain'] for t in tasks}):
        ts=[t for t in per_task if t['domain']==domain]
        per_domain.append(dict(domain=domain,tasks=len(ts),
            task_balanced_decode_ratio_128_over_64=geomean([t['decode_ratio_128_over_64'] for t in ts]),
            modes={m:aggregate([r for r in rows if r['domain']==domain]) for m,rows in allrows.items()}))
    result=dict(tasks_sha256=taskhash, modes=per_mode, per_task=per_task,per_domain=per_domain,
        task_balanced_decode_ratio_128_over_64=geomean([t['decode_ratio_128_over_64'] for t in per_task]),
        domain_balanced_decode_ratio_128_over_64=geomean([d['task_balanced_decode_ratio_128_over_64'] for d in per_domain]),
        tasks_128_faster_over_5pct=sum(t['decode_ratio_128_over_64']>1.05 for t in per_task),
        tasks_128_slower_over_5pct=sum(t['decode_ratio_128_over_64']<.95 for t in per_task),
        exact_output_match_pairs=sum(t['exact_output_match_pairs'] for t in per_task),
        complete_pairs=sum(t['complete_pairs'] for t in per_task),
        complete_pair_task_balanced_decode_ratio_128_over_64=geomean([
            t['complete_pair_decode_ratio_128_over_64'] for t in per_task if t['complete_pairs']])
            if any(t['complete_pairs'] for t in per_task) else None,
        limitations=['Convenience sample, not a measured usage-frequency distribution.',
                     'Two greedy repeats, all 64K requests before all 128K requests; order/thermal effects remain.',
                     'Different outputs can differ in length and difficulty; elapsed and token throughput are distinct.',
                     'Text and replayed context only; no live autonomous tool/browser workflow or multimodal/concurrency test.',
                     'Fixed KV pool 65536, not production automatic KV sizing.'])
    save(out/'summary.json',result)
    save(out/'rows-with-coverage.json',allrows)
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_task','per_domain']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
