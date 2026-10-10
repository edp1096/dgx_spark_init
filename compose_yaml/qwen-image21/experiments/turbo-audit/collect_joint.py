"""Report unified system memory and NVML process memory without adding them."""
import json,os
from pathlib import Path
import run as a
from summarize import rows,gpu_value
memory=rows(a.ROOT/'memory.jsonl');gpu=rows(a.ROOT/'gpu.jsonl');report={}
for profile in ['joint-baseline','joint-turbo-nvfp4','joint-turbo-selective','joint-turbo-fp8']:
 p=a.ROOT/profile
 if not (p/'ready.json').exists():continue
 m=[x for x in memory if x.get('state',{}).get('profile')==profile]
 measured=[x for x in m if x['state']['stage'] not in ('startup','warmup')]
 g=[gpu_value(x['processes']) for x in gpu if x.get('state',{}).get('profile')==profile and x['state']['stage'] not in ('startup','warmup')]
 results={f.stem:json.loads(f.read_text()) for f in (p/'results').glob('*.json') if f.stem!='warmup'}
 ready=json.loads((p/'ready.json').read_text())
 report[profile]={'cases':{n:{k:d.get(k) for k in ['status','seconds','image']} for n,d in results.items()},'resident_dits_gib':ready['resident'],'gpu_all_processes_observed_max_gib':max((x for x in g if x is not None),default=None)}
 if measured:report[profile].update(system_available_min_gib=min(x['available'] for x in measured)/2**30,immediate_free_min_gib=min(x['free'] for x in measured)/2**30)
 if m:report[profile].update(including_startup_available_min_gib=min(x['available'] for x in m)/2**30,host_swap_written_mib=max(0,m[-1]['pswpout']-m[0]['pswpout'])*os.sysconf('SC_PAGE_SIZE')/2**20,image_cgroup_swap_peak_bytes=max(x.get('cgroup',{}).get('swap',0) for x in m),image_cgroup_oom_kills=max(x.get('cgroup',{}).get('events',{}).get('oom_kill',0) for x in m))
 names={name for x in m for name in x.get('resident_cgroups',{})}
 if names:report[profile]['resident_cgroups']={name:{'swap_peak_bytes':max(x.get('resident_cgroups',{}).get(name,{}).get('swap',0) for x in m),'oom_kills':max(x.get('resident_cgroups',{}).get(name,{}).get('events',{}).get('oom_kill',0) for x in m)} for name in sorted(names)}
a.save(a.ROOT/'joint-memory-summary.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
