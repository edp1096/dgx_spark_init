"""Restart the guarded TP2 test pair; no host reboot or memory-floor override."""
import argparse,json,os,subprocess,time,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--chunk',type=int,required=True);p.add_argument('--vocab',choices=['off','ko64k'],required=True);p.add_argument('--token',required=True);a=p.parse_args()
env=os.environ.copy();env.update(QWEN_TP2_IMAGE='dgx-sglang-qwen38-fn:sm121-tp2-vocab-v1',QWEN_TP2_CHUNK=str(a.chunk),SPARKTALK_FLASH_NEXT_DRAFT_VOCAB=a.vocab)
manager=root.parents[1]/'manage_tp2.py'
subprocess.run(['python3',str(root/'capture_state.py')],check=True)
subprocess.run(['python3',str(manager),'stop'],env=env,check=True)
subprocess.run(['python3',str(manager),'start','--context','1048576','--token',a.token],env=env,check=True)
out=root/'results'/a.token;out.mkdir(exist_ok=True)
start=time.monotonic()
for i in range(1500):
 try:
  urllib.request.urlopen('http://127.0.0.1:8012/health',timeout=2).read()
  d=json.load(urllib.request.urlopen('http://127.0.0.1:8012/get_server_info',timeout=2))
  assert d['chunked_prefill_size']==a.chunk and d['context_length']==1048576
  assert bool(d['speculative_token_map'])==(a.vocab=='ko64k')
  assert d['max_total_num_tokens']>=1048576
  (out/'server-info.json').write_text(json.dumps(d,indent=2));print('READY',a.token,'load_seconds',round(time.monotonic()-start,1),flush=True);break
 except Exception:pass
 if i%20==0:
  for rank in (0,1):
   cmd=['docker','inspect',f'sglang-qwen38-fn-tp2-{rank}','--format','{{.State.Status}}']
   if rank:cmd=['ssh','-o','BatchMode=yes','edp1096@192.168.100.60',*cmd]
   state=subprocess.check_output(cmd,text=True).strip()
   if state!='running':raise RuntimeError('rank '+str(rank)+' '+state)
  print('Waiting for model',a.token,'seconds',round(time.monotonic()-start),flush=True)
 time.sleep(1)
else:raise TimeoutError('TP2 failed to become healthy')

from measure import chat
warmup=chat('Read this calibration text and reply only OK. '+(' calibration'*(a.chunk+128)),max_tokens=8)
(out/'warmup.json').write_text(json.dumps(warmup,indent=2))
print('WARMUP_COMPLETE',a.token,flush=True)
