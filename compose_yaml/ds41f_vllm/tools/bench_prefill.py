"""Measure uncached PP with identical private fixtures and reset expert slots.

Fixture JSON: [{name, expected, request: <actual chat-completion request>}].
Only hashes, answers, timing and runtime counters are written to results.
Requires DSV41_BENCH_CONTROL=1 on both serving ranks.
"""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,hashlib,json,subprocess,time,urllib.request
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixtures',type=Path,required=True)
p.add_argument('--label',required=True)
p.add_argument('--trials',type=int,default=2)
p.add_argument("--kernel-tokens",type=int,choices=(512,1024,2048),default=512)
p.add_argument("--shared-buffers",action="store_true")
p.add_argument("--routed-pipeline",action="store_true")
p.add_argument("--final-decoder-rows",action="store_true")
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
remote='/home/edp1096/workspace/dgx_spark_init/compose_yaml/ds41f_vllm/graph-control.json'
ssh=['ssh','-o','BatchMode=yes','-o','HostKeyAlias=192.168.100.60','edp1096@10.200.0.2']
for _ in range(360):
    try:
        urllib.request.urlopen('http://127.0.0.1:8010/health',timeout=2).close();break
    except OSError:time.sleep(1)
else:raise SystemExit('Model health timeout')
inspect=json.loads(subprocess.check_output(['docker','inspect','ds41-stream-0']))[0]
assert 'DSV41_BENCH_CONTROL=1' in inspect['Config']['Env']
cmd=inspect['Config']['Cmd']
batch=int(cmd[cmd.index('--max-num-batched-tokens')+1])
fixtures=json.loads(args.fixtures.read_text())
result={'final_decoder_rows':args.final_decoder_rows,'label':args.label,'max_batched_tokens':batch,'timed_trials_per_fixture':args.trials,
        'kernel_tokens':args.kernel_tokens,'shared_buffers':args.shared_buffers,'routed_pipeline':args.routed_pipeline,
        'image_id':inspect['Image'],'image_name':inspect['Config']['Image'],
        'b12x_slots_sha256':hashlib.sha256((root/'b12x_slots.py').read_bytes()).hexdigest(),
        'kv_and_expert_cache_reset_each_request':True,'runs':[]}
dest=root/'results'/(args.label+'.json')
for trial in range(-1,args.trials):
    for fixture in fixtures:
        epoch=time.time_ns()
        control=json.dumps({'epoch':epoch,'graphs':True,'expert_io':'batch_overlap','record_routes':False,'final_decoder_rows':args.final_decoder_rows,'kernel_tokens':args.kernel_tokens,'shared_buffers':args.shared_buffers,'routed_pipeline':args.routed_pipeline}).encode()
        subprocess.run(ssh+[f"cat > '{remote}.next' && mv '{remote}.next' '{remote}'"],input=control,check=True)
        staged=root/'graph-control.json.next';staged.write_bytes(control);staged.replace(root/'graph-control.json')
        request=dict(fixture['request'])
        request.update({'stream':False,'max_completion_tokens':16,'temperature':0,
                        'chat_template_kwargs':{'thinking':False},'cache_salt':f'{args.label}-{epoch}'})
        request.pop('reasoning_effort',None);request.pop('stream_options',None)
        digest=hashlib.sha256(json.dumps(fixture['request'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        start=time.monotonic()
        req=urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=900) as response:reply=json.load(response)
        elapsed=time.monotonic()-start
        usage=reply['usage'];metrics=reply['metrics'];choice=reply['choices'][0]
        row={'fixture':fixture['name'],'trial':trial,'warmup':trial<0,'request_sha256':digest,
             'text':choice['message']['content'],'expected':fixture['expected'],'finish':choice['finish_reason'],
             'elapsed_seconds':elapsed,'usage':usage,'metrics':metrics,
             'pp':usage['prompt_tokens']/(metrics['time_to_first_token_ms']/1000),
             'ttft_seconds':(metrics['time_to_first_token_ms']+metrics['queue_time_ms'])/1000,'ranks':{}}
        for rank in (0,1):
            command=['docker','logs','--since','20m',f'ds41-stream-{rank}']
            log=subprocess.check_output(command if rank==0 else ssh+command,stderr=subprocess.STDOUT).decode()
            # Node wall clocks can differ. Select the exact reset epoch rather
            # than truncating a worker log at a timestamp from the head clock.
            lines=log.splitlines();marker='EXPERT_CACHE_RESET '+str(epoch)
            starts=[i for i,line in enumerate(lines) if marker in line]
            assert starts,(rank,epoch)
            log='\n'.join(lines[starts[-1]:])+'\n'
            (root/'results'/f'{args.label}-{fixture["name"]}-{trial}-rank{rank}.log').write_text(log)
            counters=[json.loads(line.split('EXPERT_STREAM ',1)[1]) for line in log.splitlines() if 'EXPERT_STREAM {' in line]
            memory=Path('/proc/meminfo').read_text() if rank==0 else subprocess.check_output(ssh+['cat','/proc/meminfo']).decode()
            available=int(next(line.split()[1] for line in memory.splitlines() if line.startswith('MemAvailable:')))/1048576
            row['ranks'][str(rank)]={'host_available_gib':available,'last_periodic_counters':counters[-1] if counters else None}
        result['runs'].append(row);dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('ranks','request_sha256','metrics')},ensure_ascii=False),flush=True)
        assert usage['prompt_tokens_details']['cached_tokens']==0,row
        assert row['text'].strip().strip('`". *')==row['expected'] and row['finish']=='stop',row
print('PREFILL_BENCHMARK_PASS',flush=True)
