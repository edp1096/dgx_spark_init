"""Idle-only same-process next-chunk Engram prefetch A/B, with expert/KV state controlled."""
import argparse,hashlib,json,re,time
from pathlib import Path
import requests
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--trials',type=int,default=3)
p.add_argument('--backends',default='off,on')
p.add_argument('--scheduler-label',type=int,default=4096)
p.add_argument('--require-dense-profile',action='store_true')
p.add_argument('--long-only',action='store_true')
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
backends=a.backends.split(',');assert backends and len(set(backends))==len(backends) and set(backends)<= {'off','on'}
base='http://127.0.0.1:8010'

def metrics():
 r=requests.get(base+'/metrics',timeout=10);r.raise_for_status()
 text=r.text
 values=re.findall(r'^vllm:num_requests_(?:running|waiting)\{[^\n]*\} ([0-9.eE+-]+)$',text,re.M)
 assert values and all(float(v)==0 for v in values),'Concurrent inference detected'
 return sum(float(x) for x in re.findall(r'^vllm:request_success_total\{[^\n]*\} ([0-9.eE+-]+)$',text,re.M))
def disk_io():
 import subprocess,shlex
 values=[]
 for rank in (0,1):
  cmd=['docker','exec',f'ds41-stream-{rank}','cat','/sys/fs/cgroup/io.stat']
  if rank:cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','edp1096@192.168.100.60',shlex.join(cmd)]
  r=subprocess.run(cmd,capture_output=True,text=True,check=True)
  values.append({line.split()[0]:{k:int(v) for k,v in (field.split('=') for field in line.split()[1:])} for line in r.stdout.splitlines()})
 return values

def cold_engram_pages():
 import subprocess,shlex
 code="""import os,json
 from pathlib import Path
 root=Path(os.environ.get('DSV41_ENGRAM_DIR') or os.environ['DSV41_MODEL'])
 index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
 names={v for k,v in index.items() if '.engram.embed.' in k}
 assert names
 for name in names:
  fd=os.open(root/name,os.O_RDONLY)
  try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally:os.close(fd)
 print(len(names))
 """
 import textwrap
 code=textwrap.dedent(code.replace('import os,json\n',' import os,json\n',1))
 for rank in (0,1):
  cmd=['docker','exec',f'ds41-stream-{rank}','python3','-c',code]
  if rank:cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','edp1096@192.168.100.60',shlex.join(cmd)]
  r=subprocess.run(cmd,capture_output=True,text=True,check=True)
  assert r.stdout.strip()=='2',r.stdout

def rpc(method,args):
 r=requests.post(base+'/collective_rpc',json={'method':method,'args':args,'timeout':180},timeout=200)
 assert r.ok,(r.status_code,r.text[:1200])
 results=r.json()['results'];assert len(results)==2,results;return results
for _ in range(300):
 try:
  if requests.get(base+'/health',timeout=2).ok:break
 except requests.RequestException:pass
 time.sleep(2)
else:raise RuntimeError('Model health timeout')
fixtures=[] if a.long_only else json.loads((Path.home()/'.local/state/ds41-probes/20260912-retry4096/fixtures.json').read_text())
target='\n'.join(f'Item {i:03d}: preserve original weights and verify the result.' for i in range(48))
fixtures.append({'name':'decode-copy-48','expected':target,'request':{'messages':[{'role':'user','content':'Copy the following lines exactly. Return only those lines, without code fences.\n'+target}]},'max_tokens':1024})
if a.long_only:
 text='\n'.join(f'Record {i:04d}: key={hashlib.sha256(str(i).encode()).hexdigest()[:24]}; value={i*7919%104729}; category=item-{i%37}.' for i in range(850))
 text+='\nThe verification phrase is silver-forest-827. Return only that verification phrase.'
 fixtures=[{'name':'unique-records-long','expected':'silver-forest-827','request':{'messages':[{'role':'user','content':text}]},'max_tokens':32}]
result={'engram_cache':'cold-advisory' if a.long_only else 'warm-repeat','prefetch_only_change':len(backends)==2,'scheduler_label':a.scheduler_label,'dense_profile_required':a.require_dense_profile,'preload_count':224,'trials':a.trials,'runs':[]};a.output.write_text(json.dumps(result))
for trial in range(-1,a.trials):
 for fixture in fixtures:
  order=backends if trial%2 else backends[::-1]
  pair={}
  for backend in order:
   before=metrics();rpc('ds41_engram_benchmark',['native']);setting=rpc('ds41_engram_prefetch_benchmark',[backend])
   if a.require_dense_profile:
    profiles=rpc('ds41_dense_profile_status',[])
    assert all(x['matched']==x['expected']==8 and not x['winner_conflicts'] for x in profiles),profiles
   assert all(x['enabled']==(backend=='on') for x in setting)
   seed=rpc('ds41_preload_benchmark',['seed','224','0']);assert all(x['count']==224 for x in seed)
   stats0=rpc('ds41_preload_benchmark',['stats','0','0'])
   if a.long_only:cold_engram_pages()
   io_before=disk_io() if a.long_only else None
   body=dict(fixture['request']);body.update(model='deepseek-v4.1-flash',stream=False,temperature=0,seed=42,max_completion_tokens=fixture.get('max_tokens',32),chat_template_kwargs={'thinking':False},cache_salt=f'engram-{time.time_ns()}')
   body.pop('reasoning_effort',None);body.pop('stream_options',None)
   start=time.perf_counter();r=requests.post(base+'/v1/chat/completions',json=body,timeout=900);r.raise_for_status();reply=r.json();elapsed=time.perf_counter()-start
   assert metrics()-before==1,'Foreign request during measurement'
   io_after=disk_io() if a.long_only else None
   after=rpc('ds41_preload_benchmark',['stats','0','0']);prefetch=rpc('ds41_engram_prefetch_benchmark',['stats'])
   assert all(x['errors']==0 for x in prefetch),prefetch
   c=reply['choices'][0];actual=c['message']['content'];expected=fixture['expected']
   assert actual.strip().strip('`\". *')==expected.strip().strip('`\". *') and c['finish_reason']=='stop',(fixture['name'],backend,actual)
   usage=reply['usage'];m=reply['metrics'];assert usage['prompt_tokens_details']['cached_tokens']==0,usage
   pair[backend]=actual
   row={'trial':trial,'warmup':trial<0,'fixture':fixture['name'],'backend':backend,'seconds':elapsed,'usage':usage,'metrics':m,'exact':True,'request_sha256':hashlib.sha256(json.dumps(fixture['request'],sort_keys=True).encode()).hexdigest(),'output_sha256':hashlib.sha256(actual.encode()).hexdigest(),'ttft_s':(m['time_to_first_token_ms']+m['queue_time_ms'])/1000,'tg':(usage['completion_tokens']-1)/(m['generation_time_ms']/1000),'prefetch':prefetch,'io_before':io_before,'io_after':io_after,'expert_delta':[{k:y[k]-x[k] for k in ('slot_hits','slot_misses','packed_read_bytes','demand_read_bytes')} for x,y in zip(stats0,after)]}
   result['runs'].append(row);a.output.write_text(json.dumps(result,indent=2)+'\n')
   print(json.dumps({k:row[k] for k in ('trial','fixture','backend','ttft_s','tg','exact')}),flush=True)
  assert len(set(pair.values()))==1,'Reader variants changed output bytes'
print('PASS',flush=True)
