#!/usr/bin/env python3
"""Manage the separate, guarded SGLang TP2 deployment on two Sparks."""
import argparse,datetime,ipaddress,json,os,re,shlex,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORKER=os.environ.get('QWEN_TP2_WORKER','edp1096@192.168.100.60')
WORKER_ROOT=Path(os.environ.get('QWEN_TP2_WORKER_ROOT',str(ROOT)))
def name(rank):return os.environ.get('QWEN_TP2_HEAD_CONTAINER' if rank==0 else 'QWEN_TP2_WORKER_CONTAINER',f'sglang-qwen38-fn-tp2-{rank}')
def node_root(rank):return ROOT if rank==0 else WORKER_ROOT
IMAGE=os.environ.get('QWEN_TP2_IMAGE','dgx-sglang-qwen38-fn:sm121-b12x-head-v1')
def run(rank,args,**kw):
    args=[str(a).replace(str(ROOT),str(WORKER_ROOT)) for a in args] if rank else args
    cmd=args if rank==0 else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',WORKER,shlex.join(args)]
    return subprocess.run(cmd,check=True,**kw)
def output(rank,args):return run(rank,args,capture_output=True,text=True).stdout
def inspect(rank,name):
    try:return json.loads(output(rank,['docker','inspect',name]))[0]
    except subprocess.CalledProcessError:return None
def gid(rank):
    address=os.environ.get('QWEN_TP2_HEAD','10.200.0.1') if rank==0 else os.environ.get('QWEN_TP2_WORKER_RAIL','10.200.0.2')
    code='''from pathlib import Path
import ipaddress
root=Path('/sys/class/infiniband')/HCA/'ports/1'
for p in (root/'gids').iterdir():
 try:
  ip=ipaddress.ip_address(p.read_text().strip())
  if str(ip.ipv4_mapped)==ADDRESS and 'v2' in (root/'gid_attrs/types'/p.name).read_text():
   print(p.name);break
 except ValueError:pass
else:raise RuntimeError('No matching RoCEv2 GID')
'''.replace('ADDRESS',repr(address)).replace('HCA',repr(os.environ.get('HEAD_NCCL_HCA' if rank==0 else 'WORKER_NCCL_HCA','rocep1s0f1')))
    return output(rank,['python3','-c',code]).strip()
def env_for(rank,context,token):
    env={'QWEN_TP2_RANK':str(rank),'QWEN_TP2_CONTEXT':str(context),'QWEN_TP2_TOKEN':token,'QWEN_TP2_GID':gid(rank),'QWEN_TP2_IMAGE':IMAGE}
    for key in ('QWEN_TP2_HEAD','QWEN_TP2_API_PORT','QWEN_TP2_DIST_PORT','QWEN_TP2_CHUNK','SPARKTALK_FLASH_NEXT_DRAFT_VOCAB'):
        if key in os.environ:env[key]=os.environ[key]
    env.update(QWEN_TP2_CONTAINER=name(rank),QWEN_TP2_HF_CACHE=os.environ.get('HF_CACHE' if rank==0 else 'WORKER_HF_CACHE',str(Path.home()/'.cache/huggingface')),QWEN_TP2_CACHE_ROOT=os.environ.get('QWEN_TP2_HEAD_CACHE' if rank==0 else 'QWEN_TP2_WORKER_CACHE',str(Path.home()/'.local/share/sparktalk/cache/sglang-flash-next-tp2')),QWEN_TP2_IF=os.environ.get('HEAD_NCCL_IF' if rank==0 else 'WORKER_NCCL_IF','enp1s0f1np1'),QWEN_TP2_HCA=os.environ.get('HEAD_NCCL_HCA' if rank==0 else 'WORKER_NCCL_HCA','rocep1s0f1'),QWEN_TP2_BIND=os.environ.get('QWEN_TP2_BIND','0.0.0.0'),QWEN_TP2_MODEL=os.environ.get('QWEN_TP2_MODEL','edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4'))
    return env
def start_watch(rank,token):
    code=f'''from pathlib import Path
import subprocess,json
out=Path.home()/'.local/state/qwen38-tp2'/{token!r};out.mkdir(parents=True,exist_ok=True)
with (out/'watchdog-{rank}.log').open('ab') as log:
 p=subprocess.Popen(['python3',{str(ROOT/'tp2/watchdog.py')!r},'--rank',{str(rank)!r},'--token',{token!r},'--container',{name(rank)!r}],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
'''
    run(rank,['python3','-c',code])
    ready=str(Path.home()/'.local/state/qwen38-tp2'/token/f'ready-rank{rank}.json')
    for _ in range(40):
        try:output(rank,['test','-f',ready]);return
        except subprocess.CalledProcessError:time.sleep(.25)
    raise RuntimeError('Watchdog did not become ready')
def stop():
    for rank in (0,1):
        container=name(rank);d=inspect(rank,container)
        if not d:continue
        token=d['Config'].get('Labels',{}).get('qwen.tp2.probe')
        if not token:raise RuntimeError('Refusing to stop an unlabelled container')
        if d['State']['Running']:run(rank,['docker','stop','-t','30',d['Id']])
        run(rank,['python3','-c',f"from pathlib import Path;p=Path.home()/'.local/state/qwen38-tp2'/{token!r};(p/'stop-rank{rank}').touch()"])
def start(context,token):
    for rank in (0,1):
        run(rank,['python3','-c',f"from pathlib import Path;p=Path.home()/'.local/state/qwen38-tp2'/{token!r};assert not p.exists(),'Choose a fresh --token for each start'"])
    # Refuse replacement of any known live LLM; user controls model selection.
    for rank in (0,1):
        names=['sglang-qwen38-fn','vllm-qwen38-fn',name(rank),f'ds41-stream-{rank}',
               'glm53-head' if rank==0 else 'glm53-worker','deepseek-v4-flash-vllm-dspark-1']
        for container in names:
            d=inspect(rank,container)
            if d and d['State']['Running']:raise RuntimeError(f'Rank {rank}: stop {container} first')
    ids=[output(rank,['docker','image','inspect',IMAGE,'--format','{{.Id}}']).strip() for rank in (0,1)]
    if len(set(ids))!=1:raise RuntimeError('Peer images differ')
    run(1,['mkdir','-p',str(ROOT)])
    subprocess.run(['rsync','-a','--exclude','bench/results/','--exclude','__pycache__/',str(ROOT)+'/',WORKER+':'+str(WORKER_ROOT)+'/'],check=True)
    for rank in (0,1):
        run(rank,['docker','run','--rm','--privileged','--pid=host','alpine:3.22','sh','-c','sync; echo 3 > /proc/sys/vm/drop_caches'])
        code="from pathlib import Path;v=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024;print(v);assert v>=110*2**30,'Need 110 GiB available before TP2 start'"
        run(rank,['python3','-c',code])
    try:
        for rank in (0,1):start_watch(rank,token)
        for rank in (1,0):
            env=env_for(rank,context,token)
            run(rank,['env',*[k+'='+v for k,v in env.items()],'docker','compose','-f',str(ROOT/'compose.tp2.yaml'),'up','-d','--no-build'])
    except Exception:
        stop()
        for rank in (0,1):
            run(rank,['python3','-c',f"from pathlib import Path;p=Path.home()/'.local/state/qwen38-tp2'/{token!r};p.mkdir(parents=True,exist_ok=True);(p/'stop-rank{rank}').touch()"])
        raise
    (Path.home()/'.local/state/qwen38-tp2/last-start.json').write_text(json.dumps({'token':token,'context':context,'image':ids[0]},indent=2)+'\n')
    print('Started guarded TP2:',token,'context',context,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('start','stop','status'))
    p.add_argument('--context',type=int,choices=(262144,524288,1048576),default=262144)
    p.add_argument('--token',default=datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.token):raise ValueError('Invalid probe token')
    if a.action=='stop':stop()
    elif a.action=='start':start(a.context,a.token)
    else:
        for rank in (0,1):
            d=inspect(rank,name(rank))
            print(rank,d['State'] if d else 'absent')
