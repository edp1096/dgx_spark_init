"""Summarize measured receipts without adding overlapping memory categories."""
import json
from pathlib import Path
import re
import statistics
import os
import trial

GIB=2**30


def read(path):return json.loads(path.read_text())


def summarize(label):
    p=trial.OUT/label
    bench=read(p/'bench.json')
    info=read(p/'server-info-final.json')
    memory=info['internal_states'][0]['qad_memory']
    rows=[json.loads(line) for line in (p/'memory.jsonl').read_text().splitlines()]
    source=(p/'docker.log').read_text()
    progress=[dict(shard=int(a),total=int(b),seconds=float(c)) for a,b,c in re.findall(
        r'SGLANG_WEIGHT_PROGRESS current=(\d+) total=(\d+) elapsed_seconds=([\d.]+)',source)]
    result=dict(label=label,context=memory['context_tokens'],actual_kv_capacity=memory['capacity_tokens'],
                cuda=memory,startup=info['startup_time'],
                decode_medians={kind:statistics.median(x['decode_tps'] for x in bench if x['name'].startswith(kind+'_'))
                                for kind in ['korean','code']},
                long_requests=[{k:x[k] for k in ['name','usage','ttft','decode_tps','seconds']} for x in bench if x['name'].startswith(('fresh_','cached_'))],
                basic_quality=dict(passed=sum(x.get('pass',False) for x in bench),total=sum('pass' in x for x in bench)),
                min_available_gib=min(x['host']['MemAvailable'] for x in rows)/GIB,
                min_free_gib=min(x['host']['MemFree'] for x in rows)/GIB,
                host_swap_io_mib={k:(rows[-1]['swap_io_pages'][k]-rows[0]['swap_io_pages'][k])*os.sysconf('SC_PAGE_SIZE')/2**20 for k in ['pswpin','pswpout']},
                container_swap_peak_bytes=max(int(x.get('cgroup',{}).get('memory.swap.current',0)) for x in rows),
                container_oom=any(x['oom'] or re.search(r'oom(?:_kill)? [1-9]',x.get('cgroup',{}).get('memory.events','')) for x in rows),
                expert_shards_seconds=next((x['seconds'] for x in progress if x['total']==206 and x['shard']==192),None),
                fp8_modules=source.count('RADIXARK_FP8_LAYER '),fp8_verified_modules=source.count('RADIXARK_FP8_VERIFIED '))
    for kind,filename,key in [('extended_quality','quality.json','pass'),('thinking_quality','thinking-quality.json','pass_')]:
        if (p/filename).exists():
            items=read(p/filename);result[kind]=dict(passed=sum(x[key] for x in items),total=len(items),failed=[x['name'] for x in items if not x[key]])
    result['literal_xml']=next({k:x[k] for k in ['text','calls','finish']} for x in bench if x['name']=='literal_exact_xml')
    if (p/'resident.json').exists():result['residency']=read(p/'resident.json')
    for size in [1024,4096]:
        a=next(x for x in bench if x['name']==f'fresh_{size}')
        b=next(x for x in bench if x['name']==f'cached_{size}')
        result.setdefault('prefix_cache',[]).append(dict(prompt_tokens=a['usage']['prompt_tokens'],identical_text=a['text']==b['text'],fresh_ttft=a['ttft'],cached_ttft=b['ttft']))
    return result


if __name__=='__main__':
    results={label:summarize(label) for label in ['baseline-a','baseline-opt','hybrid-opt']}
    a=results['baseline-opt'];b=results['hybrid-opt']
    delta=dict(cuda_allocated_saved_gib=a['cuda']['cuda_allocated_gib']-b['cuda']['cuda_allocated_gib'],
               decode_percent={k:(b['decode_medians'][k]/a['decode_medians'][k]-1)*100 for k in ['korean','code']},
               startup_seconds_saved=results['baseline-a']['startup']['tokenizer_e2e']-a['startup']['tokenizer_e2e'])
    parser={name:{k:v for k,v in read(trial.OUT/(name+'.json')).items() if k!='results'}
            for name in ['parser-original','parser-patched','reasoning-original','reasoning-patched']}
    output=dict(models=results,delta=delta,parser=parser,conversion=read(trial.OUT/'conversion-receipt.json'),
                assessment=read(trial.OUT/'assessment.json'),velo_code_review=read(trial.OUT/'velo-code-review.json') if (trial.OUT/'velo-code-review.json').exists() else read(trial.OUT.parent/'expert-fast-path-review-20261009/velo-code-review.json'),
                caveats=['One startup per configuration; the first startup overlapped the initial small GPU linear probes.',
                         'Decode rates are medians of three requests per prompt; output text can differ between quantizations.',
                         '57K prompt tested; 1M is verified KV capacity, not a full 1M quality test.',
                         'CUDA allocator, driver, cgroup and host available memory overlap; do not add them.',
                         'Host swap IO is measured system-wide; container swap is tracked separately.'])
    (trial.OUT/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(dict(delta=delta,models={k:{f:v[f] for f in ['decode_medians','basic_quality','min_available_gib','host_swap_io_mib','fp8_verified_modules']} for k,v in results.items()}),ensure_ascii=False,indent=2))
