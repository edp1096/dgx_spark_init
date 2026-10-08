"""Idle-only same-process Engram reader A/B, with expert/KV state controlled."""
import argparse,hashlib,json,re,time
from pathlib import Path
import requests
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--trials',type=int,default=3)
p.add_argument('--backends',default='python,native')
p.add_argument('--scheduler-label',type=int,default=4096)
p.add_argument('--require-dense-profile',action='store_true')
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
backends=a.backends.split(',');assert backends and len(set(backends))==len(backends) and set(backends)<= {'python','native'}
base='http://127.0.0.1:8010'

def metrics():
 r=requests.get(base+'/metrics',timeout=10);r.raise_for_status()
 text=r.text
 values=re.findall(r'^vllm:num_requests_(?:running|waiting)\{[^\n]*\} ([0-9.eE+-]+)$',text,re.M)
 assert values and all(float(v)==0 for v in values),'Concurrent inference detected'
 return sum(float(x) for x in re.findall(r'^vllm:request_success_total\{[^\n]*\} ([0-9.eE+-]+)$',text,re.M))
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
fixtures=json.loads((Path.home()/'.local/state/ds41-probes/20260912-retry4096/fixtures.json').read_text())
target='\n'.join(f'Item {i:03d}: preserve original weights and verify the result.' for i in range(48))
fixtures.append({'name':'decode-copy-48','expected':target,'request':{'messages':[{'role':'user','content':'Copy the following lines exactly. Return only those lines, without code fences.\n'+target}]},'max_tokens':1024})
result={'reader_only_change':len(backends)==2,'scheduler_label':a.scheduler_label,'dense_profile_required':a.require_dense_profile,'preload_count':224,'trials':a.trials,'runs':[]};a.output.write_text(json.dumps(result))
for trial in range(-1,a.trials):
 for fixture in fixtures:
  order=backends if trial%2 else backends[::-1]
  pair={}
  for backend in order:
   before=metrics();setting=rpc('ds41_engram_benchmark',[backend])
   if a.require_dense_profile:
    profiles=rpc('ds41_dense_profile_status',[])
    assert all(x['matched']==x['expected']==8 and not x['winner_conflicts'] for x in profiles),profiles
   assert all(x['backend']==backend for x in setting)
   seed=rpc('ds41_preload_benchmark',['seed','224','0']);assert all(x['count']==224 for x in seed)
   stats0=rpc('ds41_preload_benchmark',['stats','0','0'])
   body=dict(fixture['request']);body.update(model='deepseek-v4.1-flash',stream=False,temperature=0,seed=42,max_completion_tokens=fixture.get('max_tokens',32),chat_template_kwargs={'thinking':False},cache_salt=f'engram-{time.time_ns()}')
   body.pop('reasoning_effort',None);body.pop('stream_options',None)
   start=time.perf_counter();r=requests.post(base+'/v1/chat/completions',json=body,timeout=900);r.raise_for_status();reply=r.json();elapsed=time.perf_counter()-start
   assert metrics()-before==1,'Foreign request during measurement'
   after=rpc('ds41_preload_benchmark',['stats','0','0'])
   c=reply['choices'][0];actual=c['message']['content'];expected=fixture['expected']
   assert actual.strip().strip('`\". *')==expected.strip().strip('`\". *') and c['finish_reason']=='stop',(fixture['name'],backend,actual)
   usage=reply['usage'];m=reply['metrics'];assert usage['prompt_tokens_details']['cached_tokens']==0,usage
   pair[backend]=actual
   row={'trial':trial,'warmup':trial<0,'fixture':fixture['name'],'backend':backend,'seconds':elapsed,'usage':usage,'metrics':m,'exact':True,'request_sha256':hashlib.sha256(json.dumps(fixture['request'],sort_keys=True).encode()).hexdigest(),'output_sha256':hashlib.sha256(actual.encode()).hexdigest(),'ttft_s':(m['time_to_first_token_ms']+m['queue_time_ms'])/1000,'tg':(usage['completion_tokens']-1)/(m['generation_time_ms']/1000),'expert_delta':[{k:y[k]-x[k] for k in ('slot_hits','slot_misses','packed_read_bytes','demand_read_bytes')} for x,y in zip(stats0,after)]}
   result['runs'].append(row);a.output.write_text(json.dumps(result,indent=2)+'\n')
   print(json.dumps({k:row[k] for k in ('trial','fixture','backend','ttft_s','tg','exact')}),flush=True)
  assert len(set(pair.values()))==1,'Reader variants changed output bytes'
print('PASS',flush=True)
