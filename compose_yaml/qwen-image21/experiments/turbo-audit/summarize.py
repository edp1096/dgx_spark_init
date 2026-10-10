"""Summaries use identical fresh-prompt cases and separately label warm reuse."""
import argparse,json,os,statistics
from pathlib import Path
PROFILES=['baseline','turbo-nvfp4','turbo-selective','turbo-bf16','turbo-fp8']
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def gpu_value(s):
 try:return sum(float(v.split(',')[1]) for v in s.splitlines() if v.strip())/1024
 except (ValueError,IndexError):return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root;mem=rows(root/'memory.jsonl');gpu=rows(root/'gpu.jsonl');report={}
 for profile in PROFILES:
  p=root/profile
  if not p.exists():continue
  results={f.stem:json.loads(f.read_text()) for f in (p/'results').glob('*.json')}
  ready_time=json.loads((p/'ready.json').read_text())['time'] if (p/'ready.json').exists() else 0
  cold=[d for name,d in results.items() if name!='warmup' and not name.startswith('warm-') and not d.get('request',{}).get('reference_files') and d.get('status')=='success']
  warm=[d for name,d in results.items() if name in ('warm-1','warm-2') and d.get('status')=='success']
  samples=[x for x in mem if x['wall']>=ready_time and x.get('state',{}).get('profile')==profile and x.get('state',{}).get('stage') not in ('startup','warmup')]
  gpu_samples=[gpu_value(x['processes']) for x in gpu if x['wall']>=ready_time and x.get('state',{}).get('profile')==profile and x.get('state',{}).get('stage') not in ('startup','warmup')];gpu_samples=[x for x in gpu_samples if x is not None]
  if cold:
   out={'fresh_prompt_median_seconds':statistics.median(x['seconds'] for x in cold),'fresh_prompt_case_count':len(cold),'sample_median_seconds':statistics.median(x['stages']['sample'] for x in cold),'warm_prompt_median_seconds':statistics.median(x['seconds'] for x in warm) if warm else None,'steps':cold[0]['sampling_steps'],'cuda_torch_allocated_peak_gib':max(x.get('peak_cuda_allocated_gib',0) for x in results.values()),'gpu_process_peak_gib':max(gpu_samples,default=0),'case_seconds':{n:d.get('seconds') for n,d in results.items() if n!='warmup'},'image_checks':{n:d.get('image') for n,d in results.items() if n!='warmup'}}
   if samples:
    out.update(system_available_min_gib=min(x['available'] for x in samples)/2**30,immediate_free_min_gib=min(x['free'] for x in samples)/2**30,host_swap_written_mib=max(0,samples[-1]['pswpout']-samples[0]['pswpout'])*os.sysconf('SC_PAGE_SIZE')/2**20,cgroup_swap_peak_bytes=max(x.get('cgroup',{}).get('swap',0) for x in samples),cgroup_oom_events=max(x.get('cgroup',{}).get('events',{}).get('oom_kill',0) for x in samples))
   idle=[json.loads(s) for s in (p/'events.jsonl').read_text().splitlines() if json.loads(s)['event']=='idle']
   if idle:out['idle_cuda_allocated_gib']=idle[-1]['allocated_GiB'];out['resident_dit_gib']=idle[-1]['resident']['qwim']
   report[profile]=out
 baseline=report.get('baseline',{})
 baseline_results={f.stem:json.loads(f.read_text()) for f in (root/'baseline/results').glob('*.json')}
 for name,out in report.items():
  if baseline:
   own={f.stem:json.loads(f.read_text()) for f in (root/name/'results').glob('*.json')}
   common=[n for n,d in own.items() if n in baseline_results and n!='warmup' and not n.startswith('warm-') and not d.get('request',{}).get('reference_files') and d.get('status')=='success']
   out['speedup_common_cases']=sorted(common)
   out['fresh_prompt_speedup_vs_baseline']=statistics.median(baseline_results[n]['seconds'] for n in common)/statistics.median(own[n]['seconds'] for n in common)
   out['sampling_speedup_vs_baseline']=statistics.median(baseline_results[n]['stages']['sample'] for n in common)/statistics.median(own[n]['stages']['sample'] for n in common)
 (root/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps({name:{k:v for k,v in x.items() if k not in ('case_seconds','image_checks')} for name,x in report.items()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
