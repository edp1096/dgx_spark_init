#!/usr/bin/env python3
"""Isolated worker-only test. Never replaces an existing model container."""
import argparse,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--token',required=True);a=p.parse_args()
root=pathlib.Path.home();name='huihui-validation';state=root/'.local/state/qwen38-tp2'/a.token
assert not state.exists();assert not subprocess.check_output(['docker','ps','-q']).strip(),'Worker has running containers'
mem=int(next(s.split()[1] for s in pathlib.Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
assert mem>110*2**30,'Not enough free worker memory'
model=root/'.cache/huggingface'/a.model;index=json.loads((model/'model.safetensors.index.json').read_text())
assert all((model/s).is_file() for s in set(index['weight_map'].values()))
state.mkdir(parents=True);cache=root/'.local/share/huihui-validation';cache.mkdir(exist_ok=True);(cache/'ple').mkdir(exist_ok=True)
with (state/'watchdog-0.log').open('ab') as log:
 subprocess.Popen(['python3','/tmp/huihui-watchdog.py','--rank','0','--token',a.token,'--container',name],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(40):
 if (state/'ready-rank0.json').exists():break
 time.sleep(.25)
else:raise RuntimeError('Watchdog not ready')
args=json.loads(pathlib.Path(__file__).with_name('validation-args.json').read_text());args[args.index('--model-path')+1]='/hf/'+a.model
cmd=['docker','run','-d','--name',name,'--label','qwen.tp2.probe='+a.token,'--gpus','all','--network','host','--ipc','host','--memory','104g','--memory-swap','104g','--ulimit','memlock=-1','--ulimit','core=0','--cap-add','IPC_LOCK']
for k,v in {'HF_HOME':'/hf','HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','SPARKTALK_FLASH_NEXT_DRAFT_VOCAB':'off','SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB':'8','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','TORCHINDUCTOR_CACHE_DIR':'/root/.cache/sglang/inductor','MAX_JOBS':'1','TORCHINDUCTOR_COMPILE_THREADS':'4'}.items():cmd+=['-e',k+'='+v]
for source,dest,ro in [(root/'.cache/huggingface','/hf',True),(cache,'/root/.cache/sglang',False),(cache/'ple','/ple',False)]:cmd+=['--mount',f'type=bind,src={source},dst={dest}'+(',readonly' if ro else '')]
cmd+=['--entrypoint','python3','dgx-sglang-qwen38-fn:sm121-vocab1','/opt/sparktalk-flash-next/launch.py']+args
(state/'launch.json').write_text(json.dumps(cmd,indent=2))
try:subprocess.run(cmd,check=True)
except BaseException:
 (state/'stop-rank0').touch()
 raise
