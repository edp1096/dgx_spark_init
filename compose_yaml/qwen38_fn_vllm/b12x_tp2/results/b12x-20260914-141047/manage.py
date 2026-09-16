import argparse,json,subprocess,shlex,time,datetime,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
IMAGE='eugr/spark-vllm-b12x@sha256:8e7e062186f841453ef0ec6f713043c5b65447decc3835206685128c18e42262'
PEER='edp1096@192.168.100.60'
def run(rank,args,**kw):
 if rank:args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',PEER,shlex.join(args)]
 return subprocess.run(args,check=True,**kw)
def output(rank,args):return run(rank,args,capture_output=True,text=True).stdout
def name(rank):return f'qwen38-b12x-tp2-{rank}'
def info(rank):
 try:return json.loads(output(rank,['docker','inspect',name(rank)]))[0]
 except subprocess.CalledProcessError:return None

def stop():
 for rank in (0,1):
  d=info(rank)
  if d and d['State']['Running']:run(rank,['docker','stop','-t','30',d['Id']])

def start(context):
 token='b12x-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
 ids=[]
 for rank in (0,1):
  running=output(rank,['docker','ps','--format','{{.Names}}']).splitlines()
  unexpected=[x for x in running if not x.startswith('sparktalk-extra-')]
  if unexpected:raise RuntimeError(f'Rank {rank}: running model/services {unexpected}')
  ids.append(output(rank,['docker','image','inspect',IMAGE,'--format','{{.Id}}']).strip())
 if ids[0]!=ids[1]:raise RuntimeError('Images differ')
 run(1,['mkdir','-p',str(ROOT)])
 subprocess.run(['rsync','-a','--exclude','results/','--exclude','__pycache__/',str(ROOT)+'/',PEER+':'+str(ROOT)+'/'],check=True)
 for rank in (0,1):
  d=info(rank)
  if d:run(rank,['docker','rm',d['Id']])
  run(rank,['docker','run','--rm','--privileged','--pid=host','alpine:3.22','sh','-c','sync; echo 3 > /proc/sys/vm/drop_caches'])
  run(rank,['python3','-c',"from pathlib import Path;v=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024;print(v);assert v>110*2**30"])
  code=f"from pathlib import Path;import subprocess;p=Path.home()/'.local/state/qwen38-tp2'/{token!r};p.mkdir(parents=True);f=(p/'watch.log').open('ab');subprocess.Popen(['python3',{str(ROOT/'watchdog.py')!r},'--rank',{str(rank)!r},'--token',{token!r},'--container',{name(rank)!r}],stdout=f,stderr=f,start_new_session=True)"
  run(rank,['python3','-c',code])
 time.sleep(1)
 try:
  for rank in (1,0):
   run(rank,['python3','-c',f"import json,os;from pathlib import Path;p=Path.home()/'.local/state/qwen38-tp2'/{token!r}/'ready-rank{rank}.json';os.kill(json.loads(p.read_text())['pid'],0)"])
   gidcode="from pathlib import Path;import ipaddress;r=Path('/sys/class/infiniband/rocep1s0f1/ports/1');print(next(p.name for p in (r/'gids').iterdir() if str(ipaddress.ip_address(p.read_text().strip()).ipv4_mapped)=='ADDRESS' and 'v2' in (r/'gid_attrs/types'/p.name).read_text()))".replace('ADDRESS','10.200.0.1' if rank==0 else '10.200.0.2')
   gid=output(rank,['python3','-c',gidcode]).strip()
   env={'RANK':str(rank),'CONTEXT':str(context),'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','VLLM_PLE_TABLE_MEMORY':'disk','VLLM_MXFP8_LM_HEAD':'0','VLLM_MTP_NVFP4_LM_HEAD':'0','CUTE_DSL_ARCH':'sm_121a','SAFETENSORS_FAST_GPU':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_SSM_CONV_STATE_LAYOUT':'DS','VLLM_USE_AOT_COMPILE':'1','VLLM_USE_MEGA_AOT_ARTIFACT':'1','VLLM_USE_V2_MODEL_RUNNER':'1','B12X_POLICY_MODE':'auto','VLLM_QWEN3_8_FLASH_NEXT_OVERLAP':'0','NCCL_SOCKET_IFNAME':'enp1s0f1np1','GLOO_SOCKET_IFNAME':'enp1s0f1np1','NCCL_IB_HCA':'rocep1s0f1','NCCL_IB_GID_INDEX':gid,'NCCL_IB_DISABLE':'0','NCCL_CUMEM_ENABLE':'0','NCCL_NVLS_ENABLE':'0','NCCL_DEBUG':'WARN','VLLM_HOST_IP':'10.200.0.1' if rank==0 else '10.200.0.2','OMP_NUM_THREADS':'4','MAX_JOBS':'1','TORCHINDUCTOR_COMPILE_THREADS':'2','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
   cache=str(Path.home()/'.local/share/sparktalk/cache/qwen38-b12x-probe')
   args=['docker','run','-d','--name',name(rank),'--network','host','--ipc','host','--security-opt','seccomp=unconfined','--gpus','all','--memory','104g','--memory-swap','104g','--ulimit','memlock=-1','--ulimit','core=0','--device','/dev/infiniband','--cap-add','IPC_LOCK','--label','qwen.tp2.probe='+token,'-v',str(Path.home()/'.cache/huggingface')+':/hf:ro','-v',str(ROOT)+':/probe:ro','-v',cache+':/root/.cache']
   for k,v in env.items():args+=['-e',k+'='+v]
   args+=['--entrypoint','python3',IMAGE,'/probe/entrypoint.py'];run(rank,args)
 except Exception:stop();raise
 (ROOT/'last-start.json').write_text(json.dumps({'token':token,'context':context,'image':ids[0],'environment':env,'async_scheduling':False,'runtime_head_requantization':False},indent=2)+'\n')
 print('Started',token,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop','status']);p.add_argument('--context',type=int,default=1048576);a=p.parse_args()
 if a.action=='start':start(a.context)
 elif a.action=='stop':stop()
 else:
  for rank in (0,1):
   d=info(rank);print(rank,d['State'] if d else None)
