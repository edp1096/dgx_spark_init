"""Summarize measured host memory and media/LLM results without hiding failures."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--candidate-root',type=Path);a=ap.parse_args()
    rows=[json.loads(x) for x in (a.root/'memory.jsonl').read_text().splitlines()]
    gpu=[json.loads(x) for x in (a.root/'gpu.jsonl').read_text().splitlines()]
    if a.candidate_root:
        for row in [*rows,*gpu]:
            if row['phase'].startswith('velo-'):row['phase']='native-swappable-control-'+row['phase']
        rows.extend(json.loads(x) for x in (a.candidate_root/'memory.jsonl').read_text().splitlines())
        gpu.extend(json.loads(x) for x in (a.candidate_root/'gpu.jsonl').read_text().splitlines())
    phases={}
    for phase in dict.fromkeys(x['phase'] for x in rows):
        rr=[x for x in rows if x['phase']==phase]
        gs=[x for x in gpu if x['phase']==phase]
        peaks=[]
        for x in gs:
            values=[int(l.rsplit(',',1)[1].strip()) for l in x['processes'].splitlines() if l.rsplit(',',1)[-1].strip().isdigit()]
            peaks.append(sum(values)/1024)
        names=set(n for x in rr for n in x['cgroups'])
        phases[phase]={'samples':len(rr),'seconds':rr[-1]['wall']-rr[0]['wall'],
            'min_available_GiB':min(x['available'] for x in rr)/2**30,'min_free_GiB':min(x['free'] for x in rr)/2**30,
            'first_available_GiB':rr[0]['available']/2**30,'last_available_GiB':rr[-1]['available']/2**30,
            'swap_written_MiB':(rr[-1]['pswpout']-rr[0]['pswpout'])*os.sysconf('SC_PAGE_SIZE')/2**20,
            'swap_read_MiB':(rr[-1]['pswpin']-rr[0]['pswpin'])*os.sysconf('SC_PAGE_SIZE')/2**20,
            'max_total_GPU_allocated_GiB':max(peaks,default=None),
            'max_container_swap_bytes':{n:max(x['cgroups'].get(n,{}).get('swap',0) for x in rr) for n in names},
            'max_velo_process_swap_bytes':max((x.get('velo',{}).get('VmSwap',0) for x in rr),default=0)}
    report={'scope':'Current Talk EXL3 bundle with cached production auxiliary images; private Talk DB/config, externally managed LLM only.',
        'root':str(a.root),'phases':phases,'runs':{},'limitations':['Measured trials, not a broad answer-quality benchmark.','Experimental Velo launch is manual; production ExLlama startup adapter is unchanged.']}
    for label in ['exllama','velo']:
        root=(a.candidate_root if label=='velo' and a.candidate_root else a.root)/label
        if not root.exists():continue
        tests={}
        for name in ['talk-chat','tts','asr','embedding','diarization','hybrid-retrieval','talk-image','talk-video','vision','schema','talk-image-after-1m','talk-image-after-full-context','tts-after-full-context','post-stress-chat']:
            p=root/(name+'.json')
            if p.exists():
                d=json.loads(p.read_text());v=d.get('result',{});ok=d['ok']
                if name.startswith(('talk-image','talk-video')):ok=ok and bool(v.get('saved'))
                if name=='vision':
                    try:ok=ok and json.loads(v['text'])==['red','green','blue','yellow']
                    except (KeyError,ValueError):ok=False
                tests[name]={'passed':ok,'wall_seconds':d['wall_seconds'],'error':d.get('error'),'response':d.get('response')}
        recalls={}
        for name in ['recall-320k','recall-512k','recall-1m']:
            p=root/(name+'.json')
            if p.exists():recalls[name]=json.loads(p.read_text())
        events=root/'media-events.jsonl'
        media=[]
        if events.exists():
            ee=[json.loads(x) for x in events.read_text().splitlines()]
            for x in ee:
                if x.get('event')=='idle':media.append({k:x[k] for k in ['case','seconds','peak_cuda_allocated_gib','peak_cuda_reserved_gib','resident'] if k in x})
        report['runs'][label]={'tests':tests,'recall':recalls,'media_completed':media}
        for name in ['managed-start.json','capacity.json','cgroup-limits.json','diarization.json','video-probe.json','weight-cache-release.json','llm-ready-memory.json','retrieval-quality.json']:
            p=root/name
            if p.exists():
                d=json.loads(p.read_text())
                if name=='managed-start.json':d={'operation':d.get('operation')}
                if name=='weight-cache-release.json' and d.get('ok'):
                    v=d.get('result',{});mem={}
                    for k in ['before','after']:
                        m={x.split(':')[0]:int(x.split()[1])*1024 for x in v[k].splitlines() if len(x.split())>=3}
                        mem[k]={key:m[key]/2**30 for key in ['MemAvailable','MemFree']}
                    d={'ok':True,'files':v['files'],'policy':v['policy'],'memory_GiB':mem}
                report['runs'][label][name]=d
    p=a.root/'results.json'
    if p.exists():report['launcher_results']=json.loads(p.read_text())
    if a.candidate_root:
        p=a.candidate_root/'results.json'
        if p.exists():report['launcher_results'].update(json.loads(p.read_text()))
        p=a.root/'velo/embedding.json'
        if p.exists():report['native_swappable_failed_control']={'embedding':json.loads(p.read_text()),'included_in_qualified_comparison':False}
    for name in ['cleanup.json','memory-floor.json']:
        p=a.root/name
        if p.exists():report[name]=json.loads(p.read_text())
        if a.candidate_root:
            p=a.candidate_root/name
            if p.exists():report['candidate_'+name]=json.loads(p.read_text())
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'phases':phases,'runs':{k:v['tests'] for k,v in report['runs'].items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
