"""Idle-only expert-preload A/B with isolated KV namespaces and repeated calls."""
import argparse, hashlib, json, re, statistics, time
from pathlib import Path
import requests

root = Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trials',type=int,default=2)
p.add_argument('--counts',default='0,128,224')
p.add_argument('--skip-warmup',action='store_true')
p.add_argument('--long-output',action='store_true')
p.add_argument('--output',type=Path,default=root/'results/expert-preload-ab.json')
a=p.parse_args();base='http://127.0.0.1:8010'

def idle():
    text=requests.get(base+'/metrics',timeout=10).text
    vals=re.findall(r'^vllm:num_requests_(?:running|waiting)\{[^\n]*\} ([0-9.eE+-]+)$',text,re.M)
    assert vals and all(float(x)==0 for x in vals),'Concurrent request detected'

def rpc(action='stats',count=0,verify=False):
    r=requests.post(base+'/collective_rpc',json={'method':'ds41_preload_benchmark','args':[action,str(count),'1' if verify else '0'],'timeout':180},timeout=200)
    r.raise_for_status();return r.json()['results']

# Existing exact-copy validation target, not part of the profile training trace.
old=json.loads((root/'results/decoder-rows-continuation.json').read_text())
target=next(x['reply']['choices'][0]['message']['content'] for x in old if x['kind']=='decode' and not x['enabled'])
copy='Copy the text between BEGIN_COPY and END_COPY verbatim. Do not add anything. Preserve existing code fences.\nBEGIN_COPY\n'+target+'\nEND_COPY\nReturn only the exact copied text.'
records='\n'.join(f'Record {i}: the sample color is blue and the value is {i%23}.' for i in range(210))
fixtures=[{'name':'short-copy','request':{'messages':[{'role':'user','content':copy}]},'expected':target},
          {'name':'long-copy','request':{'messages':[{'role':'user','content':records+'\nIgnore the records above. '+copy}]},'expected':target}]
private=Path.home()/'.local/state/ds41-probes/20260912-retry4096/fixtures.json'
fixtures.append(json.loads(private.read_text())[1])
if a.long_output:
    target='\n'.join(f'Item {i:03d}: preserve original weights and verify the result.' for i in range(48))
    prompt=records+'\nIgnore the records above. Copy every line between BEGIN_COPY and END_COPY verbatim, without code fences or extra text.\nBEGIN_COPY\n'+target+'\nEND_COPY'
    fixtures=[{'name':'long-output-copy','request':{'messages':[{'role':'user','content':prompt}]},'expected':target}]
result={'kind':'real GPU timing with both ranks and ASR running','trials':a.trials,'counts':a.counts,'same_process_warmup_reused':a.skip_warmup,'runs':[],
        'profile_sha256':hashlib.sha256((root/'expert-hot-profile.json').read_bytes()).hexdigest()}
for trial in range(0 if a.skip_warmup else -1,a.trials):
    for fixture in fixtures:
        counts=[0] if trial<0 else list(map(int,a.counts.split(',')))
        if trial%2: counts.reverse()
        for count in counts:
            idle();salt=f'preload-{time.time_ns()}'
            t=time.monotonic();seed=rpc('seed',count,False);seed_wall=time.monotonic()-t
            assert len(seed)==2 and all(x['count']==count for x in seed)
            assert seed[0]['profile_sha256']==seed[1]['profile_sha256']
            for repeat in range(1 if trial<0 or a.long_output else 2):
                idle();before=rpc()
                # Scheduler count detects a foreign request even if it completes during this call.
                metric0=requests.get(base+'/metrics',timeout=10).text
                req=dict(fixture['request']);req.update(model='deepseek-v4.1-flash',stream=False,temperature=0,seed=42,max_completion_tokens=2048 if a.long_output else 256,chat_template_kwargs={'thinking':False},cache_salt=salt)
                req.pop('reasoning_effort',None);req.pop('stream_options',None)
                start=time.monotonic();r=requests.post(base+'/v1/chat/completions',json=req,timeout=900);r.raise_for_status();reply=r.json();elapsed=time.monotonic()-start
                idle();after=rpc();choice=reply['choices'][0]
                actual=choice['message']['content'].strip()
                if len(fixture['expected'])<24: actual=actual.strip('`\". *')
                exact=actual==fixture['expected'].strip()
                assert exact and choice['finish_reason']=='stop',(fixture['name'],count,reply)
                u,m=reply['usage'],reply['metrics'];cached=u['prompt_tokens_details']['cached_tokens']
                assert cached==0 if repeat==0 else (cached>0 or u['prompt_tokens']<512),(repeat,u)
                def successes(text):return sum(float(x) for x in re.findall(r'^vllm:request_success_total\{[^\n]*\} ([0-9.eE+-]+)$',text,re.M))
                metric1=requests.get(base+'/metrics',timeout=10).text
                assert successes(metric1)-successes(metric0)==1,'Foreign request completed during measurement'
                row={'trial':trial,'warmup':trial<0,'fixture':fixture['name'],'count':count,'repeat':repeat,'seed':seed,'seed_wall_seconds':seed_wall,
                     'elapsed_seconds':elapsed,'usage':u,'metrics':m,'exact_output':exact,
                     'delta_stats':[{k:y[k]-x[k] for k in ['slot_hits','slot_misses','packed_read_bytes','demand_read_bytes','graph_replays']} for x,y in zip(before,after)],
                     'ttft_seconds':(m['time_to_first_token_ms']+m['queue_time_ms'])/1000,
                     'tg':(u['completion_tokens']-1)/(m['generation_time_ms']/1000)}
                result['runs'].append(row);a.output.write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({k:row[k] for k in ['trial','fixture','count','repeat','seed_wall_seconds','ttft_seconds','tg','exact_output']}),flush=True)
print('EXPERT_PRELOAD_AB_PASS',flush=True)
