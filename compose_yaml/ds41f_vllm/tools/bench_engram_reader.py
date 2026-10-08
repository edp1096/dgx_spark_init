"""Paired CPU Engram reader checks on real checkpoint rows, no model execution."""
import argparse,json,os,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import vllm.models.deepseek_v4_1.common.engram as e
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--rank',type=int,default=0);p.add_argument('--output',required=True)
a=p.parse_args();torch.set_num_threads(1)
root=Path(a.output);root.parent.mkdir(parents=True,exist_ok=True)
tables=[e.DiskEngramTable(a.model,layer,256,32) for layer in (1,14)]
e._kai_native_reader()
result={'rank':a.rank,'threads':e._KAI_THREADS,'chunk':e._KAI_CHUNK,'cases':[]}
gen=torch.Generator().manual_seed(20260913+a.rank)
for tokens in (1,5,6,128,1024,4096):
 # 12 lookups per token is an explicit synthetic access shape, not a routed trace.
 n=tokens*12
 requests=[]
 for table in tables:
  lo=a.rank*(table.num_rows//2);size=table.num_rows//2
  ids=torch.randint(lo,lo+size,(n,),generator=gen)
  if n>5:ids[::5]=ids[0]
  owned=torch.ones(n,dtype=torch.bool);owned[::7]=False
  requests.append((table,ids,owned))
 e._KAI_BACKEND='python';reference=e.gather_dequant_many(requests)
 e._KAI_BACKEND='native';actual=e.gather_dequant_many(requests)
 assert all(torch.equal(x,y) for x,y in zip(reference,actual)),('mismatch',tokens)
 rows=[]
 for repeat in range(6):
  for backend in (('python','native') if repeat%2==0 else ('native','python')):
   e._KAI_BACKEND=backend
   start=time.perf_counter();out=e.gather_dequant_many(requests);elapsed=time.perf_counter()-start
   assert all(torch.equal(x,y) for x,y in zip(reference,out))
   rows.append({'backend':backend,'repeat':repeat,'ms':elapsed*1000})
 summary={b:statistics.median(r['ms'] for r in rows if r['backend']==b) for b in ('python','native')}
 case={'tokens':tokens,'lookups_per_table':n,'equal':True,'median_ms':summary,'speedup':summary['python']/summary['native'],'runs':rows}
 result['cases'].append(case);root.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in case.items() if k!='runs'}),flush=True)
print('PASS',flush=True)
