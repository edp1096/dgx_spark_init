"""Aggregate qualified workload windows; never count GPU memory twice as RSS."""
import json
import os
from pathlib import Path
import statistics

ROOT=Path('/home/edp1096/.cache/model-download-jobs/velogb10-q4-vs-nvfp4-20261010')


def gpu_total(row):
    total=0
    for line in row['processes'].splitlines():
        if ',' not in line:continue
        try:total+=float(line.split(',')[1])
        except ValueError:continue
    return total/1024


def load(path):return json.loads(path.read_text())


def summary(label):
    out=ROOT/label
    start=(out/'closed-cache.json').stat().st_mtime
    end=(out/'resident-after.json').stat().st_mtime
    rows=[j for line in (ROOT/'memory.jsonl').read_text().splitlines()
          if start<=(j:=json.loads(line))['wall']<=end]
    gpu=[j for line in (ROOT/'gpu.jsonl').read_text().splitlines()
         if start<=(j:=json.loads(line))['wall']<=end]
    speed={kind:statistics.median(load(p)['result']['decode_tps'] for p in out.glob(kind+'-*.json'))
           for kind in ['korean','code']}
    checks=load(out/'coding/results.json')
    expected={'lru':15,'jsonl':9,'intervals':8,'sqlite':13,'toposort':9}
    for row in checks:
        if row['pass_']:
            execution=json.loads(row['stdout'].strip().splitlines()[-1])
            assert execution['pass'] and execution['assertions']==expected[row['name']],row
    names=set().union(*(r['cgroups'].keys() for r in rows))
    return dict(context_tokens=1048576, speed_tok_per_s=speed,
        system_available_min_GiB=min(r['available'] for r in rows)/2**30,
        immediate_free_min_GiB=min(r['free'] for r in rows)/2**30,
        total_GPU_max_sampled_GiB=max(map(gpu_total,gpu)),
        host_swap_in_MiB=(rows[-1]['pswpin']-rows[0]['pswpin'])*os.sysconf('SC_PAGE_SIZE')/2**20,
        host_swap_out_MiB=(rows[-1]['pswpout']-rows[0]['pswpout'])*os.sysconf('SC_PAGE_SIZE')/2**20,
        model_cgroup_swap_max_bytes={n:max(r['cgroups'].get(n,{}).get('swap',0) for r in rows) for n in names},
        oom_events_max={n:max(r['cgroups'].get(n,{}).get('events',{}).get('oom',0) for r in rows) for n in names},
        common_API=load(out/'common-suite/suite-summary.json'),
        coding=dict(tasks_passed=sum(r['pass_'] for r in checks),tasks_total=len(checks),assertions_total=sum(expected.values())),
        vision=load(out/'vision-quality.json'),schema=load(out/'schema.json'),
        recall=load(out/'recall-1m.json'),long_vision=load(out/'vision-long.json'),
        diarization=load(out/'diarization.json')['ok'],hybrid_retrieval=load(out/'hybrid-retrieval.json')['ok'],
        resident_ids_unchanged=load(out/'resident-before.json')==load(out/'resident-after.json'),
        window=dict(start=start,end=end),sampling=dict(host_seconds=.25,GPU_seconds=1))


if __name__=='__main__':
    result={}
    for label in ['nvfp4-1m','q4-1m']:
        if (ROOT/label/'resident-after.json').exists():result[label]=summary(label)
    (ROOT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for label,r in result.items():print(label,r['speed_tok_per_s'],r['system_available_min_GiB'],r['total_GPU_max_sampled_GiB'])
