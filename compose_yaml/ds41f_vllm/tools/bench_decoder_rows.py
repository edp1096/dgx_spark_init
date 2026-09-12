"""Interleaved, reset-cache A/B measurement of final decoder row selection."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,hashlib,json,re,subprocess,time,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixtures',type=Path,required=True)
p.add_argument('--trials',type=int,default=3)
p.add_argument("--skip-warmup",action="store_true",help="Reuse the completed same-process warmups after an interrupted run")
a=p.parse_args()
ssh=['ssh','-o','BatchMode=yes','edp1096@192.168.100.60']
inspect=json.loads(subprocess.check_output(['docker','inspect','ds41-stream-0']))[0]
assert 'DSV41_BENCH_CONTROL=1' in inspect['Config']['Env']
fixtures=json.loads(a.fixtures.read_text())
out=root/'results/decoder-rows-ab.json'
result={'image':inspect['Image'],'container':inspect['Id'],'cmd':inspect['Config']['Cmd'],
        'source_sha256':{str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest()
                        for x in [root/'decoder_rows.py',root/'streaming_moe.py',*(root/'patches').glob('*.py')]},
        'trials':a.trials,'warmup_reused_after_user_interruption':a.skip_warmup,'runs':[]}
def assert_idle():
    with urllib.request.urlopen('http://127.0.0.1:8010/metrics',timeout=10) as r:
        metrics=r.read().decode()
    values=re.findall(r'^vllm:num_requests_(?:running|waiting)\{[^\n]*\} ([0-9.eE+-]+)$',metrics,re.M)
    assert values and all(float(v)==0 for v in values),'Concurrent request detected; discard affected measurements'

for trial in range(0 if a.skip_warmup else -1,a.trials):
    for fixture in fixtures:
        for enabled in ([False,True] if trial%2 else [True,False]):
            assert_idle()
            epoch=time.time_ns()
            control={'epoch':epoch,'graphs':True,'expert_io':'batch_overlap','kernel_tokens':2048,
                     'shared_buffers':True,'final_decoder_rows':enabled,'validate_final_decoder':False}
            data=json.dumps(control).encode();path=root/'graph-control.json'
            subprocess.run(ssh+[f"cat > '{path}.next' && mv '{path}.next' '{path}'"],input=data,check=True)
            tmp=path.with_suffix('.json.next');tmp.write_bytes(data);tmp.replace(path)
            req=dict(fixture['request'])
            req.update(stream=False,max_completion_tokens=16,temperature=0,chat_template_kwargs={'thinking':False},cache_salt=f'decoder-ab-{epoch}')
            req.pop('reasoning_effort',None);req.pop('stream_options',None)
            start=time.monotonic()
            request=urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=900) as r:reply=json.load(r)
            row={'fixture':fixture['name'],'trial':trial,'warmup':trial<0,'enabled':enabled,
                 'request_sha256':hashlib.sha256(json.dumps(fixture['request'],sort_keys=True).encode()).hexdigest(),
                 'elapsed_seconds':time.monotonic()-start,'reply':reply,'ranks':{}}
            usage=reply['usage'];metrics=reply['metrics'];choice=reply['choices'][0]
            assert usage['prompt_tokens_details']['cached_tokens']==0,reply
            assert choice['finish_reason']=='stop' and choice['message']['content'].strip().strip('`". *')==fixture['expected'],reply
            row['pp']=usage['prompt_tokens']/(metrics['time_to_first_token_ms']/1000)
            row['ttft_seconds']=(metrics['time_to_first_token_ms']+metrics['queue_time_ms'])/1000
            for rank in (0,1):
                cmd=['docker','logs','--since','20m',f'ds41-stream-{rank}']
                log=subprocess.check_output(cmd if rank==0 else ssh+cmd,stderr=subprocess.STDOUT).decode().split('EXPERT_CACHE_RESET '+str(epoch),1)[1]
                (root/'results'/f'decoder-ab-{fixture["name"]}-{trial}-{int(enabled)}-rank{rank}.log').write_text(log)
                selected=[json.loads(s.split('FINAL_DECODER_ROWS ',1)[1]) for s in log.splitlines() if 'FINAL_DECODER_ROWS {' in s]
                assert bool(selected)==enabled,(rank,enabled,selected)
                assert not any('validation' in x for x in selected)
                row['ranks'][str(rank)]={'selection':selected}
            assert_idle()
            result['runs'].append(row);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({k:row[k] for k in ('fixture','trial','enabled','pp','ttft_seconds')}),flush=True)
print('DECODER_ROWS_AB_PASS',flush=True)
