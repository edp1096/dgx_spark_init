"""Summarize measured results, keeping failed/partial qualification explicit."""
import argparse,json,os,statistics
from pathlib import Path

def read(p,default=None):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return default

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root;out=r/'q5-1m'
 manifest=read(r/'manifest.json',{});memory=[json.loads(s) for s in (r/'memory.jsonl').read_text().splitlines()] if (r/'memory.jsonl').exists() else []
 gpu=[]
 if (r/'gpu.jsonl').exists():
  for s in (r/'gpu.jsonl').read_text().splitlines():
   row=json.loads(s)
   try:gpu.append(sum(float(p.split(',')[1]) for p in row['processes'].splitlines() if p.strip())/1024)
   except (ValueError,IndexError):pass
 result={'manifest':manifest,'speed':{},'quality':{},'memory':{}}
 for kind in ('code','korean'):
  rows=[read(out/(kind+'-'+str(i)+'.json'),{}) for i in range(3)]
  values=[x['result']['decode_tps'] for x in rows if x.get('ok')]
  if values:result['speed'][kind+'_median_tps']=statistics.median(values)
 suite=read(out/'common-suite/suite.json',[])
 result['quality']['common_api']={'passed':sum(bool(x.get('pass')) for x in suite if 'pass' in x),'total':sum('pass' in x for x in suite),'failures':[x.get('name') for x in suite if x.get('pass') is False]}
 for name in ('coding','vision-quality','schema','diarization','hybrid-retrieval','recall-1m-run','vision-long','post-stress-chat'):
  value=read(out/(name+'.json'))
  if value is not None:result['quality'][name]=value
 if memory:
  result['memory']={'available_min_gib':min(x['available'] for x in memory)/2**30,'free_min_gib':min(x['free'] for x in memory)/2**30,'gpu_peak_gib':max(gpu,default=0),'host_swap_written_mib':max(0,memory[-1]['pswpout']-memory[0]['pswpout'])*os.sysconf('SC_PAGE_SIZE')/2**20,'model_VmSwap_peak_gib':max((x.get('velo',{}).get('VmSwap',0) for x in memory),default=0)/2**30,'cgroups':{}}
  for x in memory:
   for name,cg in x['cgroups'].items():
    v=result['memory']['cgroups'].setdefault(name,{'current_peak_gib':0,'swap_peak_gib':0,'oom':0,'oom_kill':0})
    v['current_peak_gib']=max(v['current_peak_gib'],cg['current']/2**30);v['swap_peak_gib']=max(v['swap_peak_gib'],cg['swap']/2**30)
    for key in ('oom','oom_kill'):v[key]=max(v[key],cg['events'].get(key,0))
 result['memory_floor']=read(r/'memory-floor.json')
 (r/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({'status':manifest.get('status'),'resident_auxiliaries_verified':manifest.get('resident_auxiliaries_verified'),'speed':result['speed'],'memory':result['memory'],'common_api':result['quality']['common_api']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
