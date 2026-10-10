"""Summarize observed data, retaining the failed runs before the QSA geometry fix."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/home/edp1096/.cache/model-download-jobs/velogb10-yarn-20261009'));ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    def memory(label):
        f=a.root/label/'memory.jsonl'
        if not f.exists():return None
        rows=[json.loads(l) for l in f.read_text().splitlines()]
        gpu=[]
        for r in rows:
            total=0
            for line in r.get('gpu_processes','').splitlines():
                try:total+=int(line.split(',')[1].strip())
                except (IndexError,ValueError):pass
            gpu.append(total)
        vms=[r['process']['VmSwap'] for r in rows if 'VmSwap' in r.get('process',{})]
        cgs=[int(r['cgroup']['memory.swap.current']) for r in rows if 'memory.swap.current' in r.get('cgroup',{})]
        vm=max(vms) if vms else None
        cg=max(cgs) if cgs else None
        return {'peak_driver_gpu_GiB':max(gpu)/1024,'min_system_available_GiB':min(r['host']['MemAvailable'] for r in rows)/2**30,
                'max_process_swap_bytes':vm,'max_container_swap_bytes':cg,
                'host_swap_write_MiB':(rows[-1]['swap_io_pages']['pswpout']-rows[0]['swap_io_pages']['pswpout'])*os.sysconf('SC_PAGESIZE')/2**20,
                'observed_seconds':rows[-1]['seconds']}
    cases={}
    for label,names in {
        'native-256k-a':['recall-32k.json'],
        'yarn-1m':['recall-32k.json','recall-270k.json'],
        'yarn-1m-fresh':['recall-32k.json'],
        'yarn-64k-q8':['recall-32k.json'],
        'exllama-reference':['recall-32k.json','recall-270k.json'],
        'yarn-1m-fixed':['recall-32768.json','recall-270000.json','recall-1m.json'],
    }.items():
        cases[label]=[]
        for name in names:
            p=a.root/label/name
            if p.exists():
                j=json.loads(p.read_text());cases[label].append({'file':str(p),'prompt_tokens':j['usage']['prompt_tokens'],'output':j['text'],
                    'pass':j['pass'],'seconds':j['seconds'],'first_visible_token_seconds':j['ttft'],
                    'prefill_tps_including_delivery_overhead':j.get('prefill_tps'),'server_usage':j['usage']})
    short=a.root/'yarn-1m-fixed/responses.json'
    s=json.loads(short.read_text()) if short.exists() else []
    final=cases['yarn-1m-fixed']
    result={'source_commit':'a6ad23d60e082ff9cca2bb78adda4ecee771388c','context':1048576,'yarn_factor':4,'kv_cache':'q8','mtp_depth':3,'lanes':1,'ple':'ssd',
        'final_three_recall_cases_passed':len(final)==3 and all(r['pass'] for r in final),
        'recall_comparison':cases,'memory':{k:memory(k) for k in cases},
        'short_generation':[{'name':r['name'],'decode_tps':r.get('decode_tps'),'finish':r['finish'],'usage':r['usage']} for r in s],
        'post_full_context_reuse':json.loads((a.root/'yarn-1m-fixed/recall-after-1m.json').read_text()) if (a.root/'yarn-1m-fixed/recall-after-1m.json').exists() else None,
        'over_limit_API':json.loads((a.root/'api-boundary.json').read_text()),
        'verification':{'GPU_cases':93,'Rust_unit_tests':37,'invalid_configurations_rejected':7,'patch_exact_reproduction_files':16,
            'rope_reference_comparison':json.loads((a.root/'rope-comparison.json').read_text())},
        'failure_and_fix':'The old QSA 20-bit stride field decoded 1048577 as 1 and changed query pitch 640 to 641, corrupting sparse MTP verification. Expanding QSA geometry to 22/18/20 bits fixes this independently of the RoPE formula.',
        'limits':['Single GB10, one existing checkpoint, one lane and MTP depth 3.','Recall is a three-position synthetic test, not a broad quality benchmark.','GPU memory includes the Velo vision tower; ancillary ASR/TTS/image/embedding services were stopped.','Real TP and YaRN image-input requests were not exercised.','Reference monitoring began during startup and can miss earlier loader peaks.','Short prose/code samples were capped at 256 tokens and are speed samples, not functional code-quality tests.']}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'full_recall_complete':result['final_three_recall_cases_passed'],'final_cases':len(final),'memory':result['memory']['yarn-1m-fixed']},ensure_ascii=False))

if __name__=='__main__':main()
