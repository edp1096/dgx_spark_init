"""Monitor an already started TP2 pair and run its long-context probe."""
import argparse,importlib.util,json,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('manage',ROOT/'manage_tp2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=argparse.ArgumentParser();p.add_argument('--context',type=int,required=True);a=p.parse_args()
run=json.loads((Path.home()/'.local/state/qwen38-tp2/last-start.json').read_text());assert run['context']==a.context
out=ROOT/'bench/results'/run['token'];out.mkdir(parents=True,exist_ok=True)
name='qwen38-long-probe';proc=None

def check():
 for rank in (0,1):
  d=m.inspect(rank,f'sglang-qwen38-fn-tp2-{rank}')
  if not d or not d['State']['Running']:raise RuntimeError(f'Rank {rank} is not running: {d["State"] if d else "absent"}')
  assert d['Config']['Labels'].get('qwen.tp2.probe')==run['token'], 'Container was replaced during probe'
  path=Path.home()/'.local/state/qwen38-tp2'/run['token']
  code=f"import json,os;from pathlib import Path;p=Path({str(path)!r});r=json.loads((p/'ready-rank{rank}.json').read_text());os.kill(r['pid'],0);assert not (p/'tripped-rank{rank}.json').exists()"
  m.run(rank,['python3','-c',code],capture_output=True)

def get(path,timeout=5):
 with urllib.request.urlopen('http://127.0.0.1:8012'+path,timeout=timeout) as r:return r.read()
try:
 deadline=time.monotonic()+1200
 while True:
  check()
  try:get('/health');break
  except Exception:
   if time.monotonic()>deadline:raise TimeoutError('Startup exceeded 20 minutes')
  print('waiting for API',flush=True);time.sleep(30)
 info=json.loads(get('/get_server_info'))
 assert info['tp_size']==2 and info['nnodes']==2 and info['context_length']==a.context
 assert info['kv_cache_dtype']=='bfloat16' and not info['allow_auto_truncate']
 assert info['max_req_input_len']>=a.context-2048
 (out/'server-info.json').write_text(json.dumps(info,indent=2))
 print('API ready; starting exact-length retrieval',flush=True)
 args=['docker','run','--rm','--name',name,'--network','host','--memory','4g','--memory-swap','4g','--cpus','2','-e','HF_HUB_OFFLINE=1','-e','PYTHONUNBUFFERED=1','-v',str(Path.home()/'.cache/huggingface')+':/hf:ro','-v',str(ROOT/'tp2')+':/probe:ro','-v',str(out)+':/results','--entrypoint','python3',m.IMAGE,'/probe/long_probe.py','--context',str(a.context),'--output','/results/retrieval.json']
 with (out/'probe.log').open('w') as log:
  proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
  deadline=time.monotonic()+7200
  while proc.poll() is None:
   check()
   if time.monotonic()>deadline:raise TimeoutError('Retrieval exceeded two hours')
   print('retrieval running',flush=True);time.sleep(30)
 result=json.loads((out/'retrieval.json').read_text());print(json.dumps(result,ensure_ascii=False),flush=True)
 if proc.returncode:raise RuntimeError('Retrieval did not pass')
except BaseException:
 if proc and proc.poll() is None:subprocess.run(['docker','stop','-t','2',name],capture_output=True)
 m.stop();raise
finally:
 for rank in (0,1):
  try:
   (out/f'server-rank{rank}.log').write_text(m.output(rank,['docker','logs',f'sglang-qwen38-fn-tp2-{rank}']))
   src=str(Path.home()/'.local/state/qwen38-tp2'/run['token'])+'/'
   if rank:src=m.WORKER+':'+src
   subprocess.run(['rsync','-a',src,str(out/f'guard-rank{rank}')+'/'],check=True)
  except Exception as e:print('Evidence collection:',repr(e),flush=True)
