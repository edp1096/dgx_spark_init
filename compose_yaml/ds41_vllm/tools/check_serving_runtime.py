"""Record final normal-serving settings, long-request aftermath and memory."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,hashlib,json,shlex,subprocess,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,required=True)
p.add_argument('--kernel',type=int,default=2048);p.add_argument('--pipeline',type=int,default=0)
p.add_argument("--final-decoder-rows",type=int,choices=(0,1),default=0)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
ssh=['ssh','-o','BatchMode=yes','-o','HostKeyAlias=192.168.100.60','edp1096@10.200.0.2']
def run(rank,cmd): return subprocess.check_output(cmd if rank==0 else ssh+cmd,stderr=subprocess.STDOUT).decode()
request={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':'Reply with exactly OK.'}],
         'max_completion_tokens':8192,'temperature':0,'chat_template_kwargs':{'thinking':False}}
start=time.monotonic()
req=urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=120) as response: reply=json.load(response)
assert reply['choices'][0]['message']['content'].strip()=='OK'
assert reply['choices'][0]['finish_reason']=='stop'
result={'post_long_short_request':{'text':'OK','seconds':time.monotonic()-start,'max_completion_tokens':8192,'usage':reply['usage']},'ranks':{}}
for rank in (0,1):
    inspect=json.loads(run(rank,['docker','inspect',f'ds41-stream-{rank}']))[0]
    env=dict(e.split('=',1) for e in inspect['Config']['Env']);cmd=inspect['Config']['Cmd']
    assert inspect['State']['Running'] and not inspect['State']['OOMKilled']
    assert env['DSV41_BENCH_CONTROL']=='0'
    assert int(env.get('DSV41_FINAL_DECODER_ROWS','0'))==a.final_decoder_rows
    assert int(env['DSV41_KERNEL_TOKENS'])==a.kernel and env['DSV41_SHARED_BUFFERS']=='1'
    assert int(env['DSV41_ROUTED_PIPELINE'])==a.pipeline
    assert int(cmd[cmd.index('--max-num-batched-tokens')+1])==a.batch
    assert int(cmd[cmd.index('--max-model-len')+1])==65536
    assert '--enable-auto-tool-choice' in cmd and '--tool-call-parser' in cmd
    assert '--enable-per-request-metrics' in cmd and '--enable-prompt-tokens-details' in cmd
    info=run(rank,['cat','/proc/meminfo'])
    available=int(next(line.split()[1] for line in info.splitlines() if line.startswith('MemAvailable:')))/1048576
    memory={name:int(run(rank,['docker','exec',f'ds41-stream-{rank}','cat','/sys/fs/cgroup/memory.'+name])) for name in ('current','peak','max')}
    events=run(rank,['docker','exec',f'ds41-stream-{rank}','cat','/sys/fs/cgroup/memory.events'])
    assert all(int(line.split()[1])==0 for line in events.splitlines() if line.startswith(('oom ','oom_kill ')))
    log=run(rank,['docker','logs','--tail','200',f'ds41-stream-{rank}'])
    counters=[json.loads(line.split('EXPERT_STREAM ',1)[1]) for line in log.splitlines() if 'EXPERT_STREAM {' in line]
    assert counters and counters[-1]['gpu_cache_bytes']==87836590080
    assert counters[-1]['prefetch_issued']==0
    code="import hashlib,json;from pathlib import Path;r=Path('/opt/ds41');print(json.dumps({f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['b12x_slots.py','routed_pipeline.py','launch.sh','manage.sh','streaming_moe.py','decoder_rows.py','patches/target_model.py','patches/model_state.py','patches/vl_model.py']}))"
    command=['docker','exec',f'ds41-stream-{rank}','python3','-c',code]
    # SSH constructs one remote shell command, so quote the Python program.
    hashes=json.loads(subprocess.check_output(command if rank==0 else ssh+[shlex.join(command)]))
    result['ranks'][str(rank)]={'image_id':inspect['Image'],'image_name':inspect['Config']['Image'],'max_batched_tokens':a.batch,
                              'settings':{k:v for k,v in env.items() if k.startswith('DSV41_') and any(w in k for w in ('FINAL_DECODER','SLOTS','SCRATCH','KERNEL','SHARED_BUFFERS','ROUTED_PIPELINE','EXPERT_IO','BENCH_CONTROL','PREFETCH','MODEL_GRAPHS'))},
                              'host_available_gib':available,'cgroup_memory_bytes':memory,'memory_events':events,'counters':counters[-1],'source_sha256':hashes}
assert result['ranks']['0']['source_sha256']==result['ranks']['1']['source_sha256']
with urllib.request.urlopen('http://127.0.0.1:8585/api/config',timeout=10) as response: config=json.load(response)
result['sparktalk']={'context_tokens':config['context']['window_tokens'],'output_allowance':config['context']['output_reserve'],'reasoning_effort':config['model'].get('reasoning_effort')}
assert result['sparktalk']['context_tokens']==65536 and result['sparktalk']['output_allowance']==8192
with urllib.request.urlopen('http://127.0.0.1:8010/health',timeout=10) as response: result['health_status']=response.status
a.output.write_text(json.dumps(result,indent=2)+'\n')
print('SERVING_RUNTIME_VALIDATION_PASS',flush=True)
