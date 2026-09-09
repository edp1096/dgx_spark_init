#!/usr/bin/env python3
"""Compare saved trials without starting a server or changing response grades."""
import argparse
import json
from pathlib import Path
import statistics


def semantic(row, task):
    try:
        if 'expected_tool' in task:
            calls=list(row['tool_calls'].values())
            return len(calls)==1 and calls[0]['name']=='get_weather' and json.loads(calls[0]['arguments'])==task['expected_tool'] and row['finish_reason'] in ['stop','tool_calls']
        if task.get('review'):return None
        text=row['content'].strip()
        if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
        return json.loads(text)==task['expected'] and not row['tool_calls'] and row['finish_reason']=='stop'
    except (ValueError,KeyError):return False


def load(directory):
    read=lambda name:json.loads((directory/name).read_text())
    assert read('summary.json')['completed'] and read('cleanup.json')['exit_code']==0
    tasks={t['id']:t for t in read('tasks.json')};rows=read('responses.json')
    assert len(rows)==3*len(tasks)
    assert len({(r['id'],r['repeat']) for r in rows})==len(rows)
    observations=[json.loads(line) for line in (directory/'memory.jsonl').read_text().splitlines()]
    phases={}
    for phase in ['loading','benchmark']:
        phases[phase]={}
        for host in ['head','worker']:
            values=[r['hosts'][host] for r in observations if r['phase']==phase]
            phases[phase][host]=dict(min_available_gib=min(r['available'] for r in values)/2**30,
                max_swap_gib=max(r['swap'] for r in values)/2**30,boot_ids=sorted({r['boot'] for r in values}))
    return dict(tasks=tasks,rows=rows,containers=read('containers.json'),plan=read('plan.json'),phases=phases,
                functional_pass=sum(semantic(r,tasks[r['id']]) is True for r in rows),
                strict_pass=sum(r['passed'] is True for r in rows))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('baseline',type=Path);ap.add_argument('candidate',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();a,b=load(args.baseline),load(args.candidate)
    assert a['tasks']==b['tasks']
    varied={'DSPARK_ENABLE_DSPARK_SWA_PREFIX','DSPARK_ENABLE_DSML_RECOVERY','ADAPTIVE_SPECULATIVE_TOKENS_WINDOW'}
    assert {k:v for k,v in a['plan']['overrides'].items() if k not in varied}=={k:v for k,v in b['plan']['overrides'].items() if k not in varied}
    for host in ['head','worker']:
        assert a['containers'][host]['image_id']==b['containers'][host]['image_id']
        assert {k:v for k,v in a['containers'][host]['environment'].items() if k not in varied}=={k:v for k,v in b['containers'][host]['environment'].items() if k not in varied}
    def report(data):
        return dict(functional_pass=data['functional_pass'],strict_pass=data['strict_pass'],requests=len(data['rows']),phases=data['phases'],
            tasks={key:dict(median_elapsed_s=statistics.median(r['elapsed_s'] for r in data['rows'] if r['id']==key),
                median_ttft_s=statistics.median(r['ttft_s'] for r in data['rows'] if r['id']==key),
                total_completion_tokens=sum(r['usage'].get('completion_tokens',0) for r in data['rows'] if r['id']==key),
                total_elapsed_s=sum(r['elapsed_s'] for r in data['rows'] if r['id']==key),
                cached_tokens=[r['usage'].get('prompt_tokens_details',{}).get('cached_tokens') for r in data['rows'] if r['id']==key]) for key in data['tasks']})
    result=dict(baseline=report(a),candidate=report(b),limitations=['Sequential same-input trials; three repeats are not independent tasks.',
        'Functional tool-call scoring checks name/arguments; strict scoring also requires finish_reason=tool_calls.',
        'Prose requires a separate manual review.','Startup swap and request-phase swap are separate observations.'])
    pairs={}
    for row in a['rows']:pairs[row['id'],row['repeat']]=row
    result['functional_regressions']=[(r['id'],r['repeat']) for r in b['rows'] if semantic(pairs[r['id'],r['repeat']],a['tasks'][r['id']]) is True and semantic(r,b['tasks'][r['id']]) is False]
    result['exact_content_matches']=sum((pairs[r['id'],r['repeat']]['content'],pairs[r['id'],r['repeat']]['tool_calls'])==(r['content'],r['tool_calls']) for r in b['rows'])
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
