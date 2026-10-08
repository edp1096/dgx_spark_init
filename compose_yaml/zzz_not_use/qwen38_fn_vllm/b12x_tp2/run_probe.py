import json,time,subprocess,urllib.request
from pathlib import Path
import manage as m
run=json.loads((m.ROOT/'last-start.json').read_text());out=m.ROOT/'results'/run['token'];out.mkdir(parents=True,exist_ok=True)
(out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
for script in ('entrypoint.py','manage.py','long_probe.py'):
 (out/script).write_text((m.ROOT/script).read_text())
proc=None
base='http://127.0.0.1:8013'
def check():
 for rank in (0,1):
  d=m.info(rank)
  if not d or not d['State']['Running']:raise RuntimeError(f'Rank {rank}: {d["State"] if d else "absent"}')
  assert d['Config']['Labels']['qwen.tp2.probe']==run['token']
  code=f"import os,json;from pathlib import Path;p=Path.home()/'.local/state/qwen38-tp2'/{run['token']!r};assert not (p/'tripped-rank{rank}.json').exists();os.kill(json.loads((p/'ready-rank{rank}.json').read_text())['pid'],0)"
  m.run(rank,['python3','-c',code],capture_output=True)
def get(path,data=None):
 r=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=120) as f:
  b=f.read();return json.loads(b) if b else None
try:
 deadline=time.monotonic()+1800
 while True:
  check()
  try:
   with urllib.request.urlopen(base+'/health',timeout=3) as f:assert f.status==200
   break
  except Exception:
   if time.monotonic()>deadline:raise TimeoutError('Startup exceeded 30 minutes')
  print('Waiting for B12X API',flush=True);time.sleep(30)
 models=get('/v1/models');(out/'models.json').write_text(json.dumps(models,indent=2))
 short=get('/v1/chat/completions',{'model':'qwen38-b12x-tp2','messages':[{'role':'user','content':'Reply with just the answer: 17+25=?'}],'max_tokens':128,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
 (out/'short.json').write_text(json.dumps(short,indent=2));assert '42' in short['choices'][0]['message']['content'],short
 print('Short answer passed; starting 1M retrieval',flush=True)
 args=['docker','run','--rm','--name','qwen38-b12x-long-probe','--network','host','--memory','4g','--memory-swap','4g','--cpus','2','-e','HF_HUB_OFFLINE=1','-e','PYTHONUNBUFFERED=1','-v',str(Path.home()/'.cache/huggingface')+':/hf:ro','-v',str(m.ROOT)+':/probe:ro','-v',str(out)+':/results','--entrypoint','python3',m.IMAGE,'/probe/long_probe.py','--context',str(run['context']),'--output','/results/retrieval.json']
 with (out/'probe.log').open('w') as log:
  proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);deadline=time.monotonic()+7200
  while proc.poll() is None:
   check()
   if time.monotonic()>deadline:raise TimeoutError('Long request exceeded two hours')
   print('Long retrieval running',flush=True);time.sleep(30)
 result=json.loads((out/'retrieval.json').read_text());print(json.dumps(result,ensure_ascii=False),flush=True)
 if proc.returncode:raise RuntimeError('Retrieval failed')
except BaseException:
 if proc and proc.poll() is None:subprocess.run(['docker','stop','-t','2','qwen38-b12x-long-probe'],capture_output=True)
 m.stop();raise
finally:
 for rank in (0,1):
  try:
   r=m.run(rank,['docker','logs',m.name(rank)],capture_output=True,text=True);(out/f'server-rank{rank}.log').write_text(r.stdout+r.stderr)
   src=str(Path.home()/'.local/state/qwen38-tp2'/run['token'])+'/'
   if rank:src=m.PEER+':'+src
   subprocess.run(['rsync','-a',src,str(out/f'guard-rank{rank}')+'/'],check=True)
  except Exception as e:print('Evidence collection:',str(e))
