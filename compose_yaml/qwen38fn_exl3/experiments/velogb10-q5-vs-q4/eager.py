"""Short correctness isolation: 1M, no CUDA graphs and no MTP, before another benchmark."""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('q5_runner',H/'run.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
a=q.a
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root;r.mkdir(mode=0o700,exist_ok=True)
parent=r.parent;s=json.loads((parent/'download-status.json').read_text());model=s['snapshot'];a.ROOT=r
assert not a.command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').strip()
cmd=['taskset','-c','5-9,15-19',str(q.SOURCE/'target/release/gb10_inference'),'--server','--model-dir',model,'--host','127.0.0.1','--port','19337','--model-name','q5-eager-audit','--max-seq-len','1048576','--rope-yarn-factor','4','--max-batch','1','--ple-ram','ssd','--kv-cache','q8','--exl3-mtp-k','3','--draft-confidence','0','--prefix-cache','on','--prefix-ckpt-mem-gb','1','--prefill-chunk','2048','--tune-table','off','--reasoning-effort','none','--thinking','off','--exl3-pdl','0','--exl3-mtp-head-n','0','--mem-watchdog-gb','1.7','--exl3-no-graph','--exl3-no-mtp']
(r/'command.json').write_text(json.dumps(cmd,indent=2));(r/'phase').write_text('q5-eager-startup')
mon=subprocess.Popen([sys.executable,str(H/'monitor.py'),str(r)]);v=None;results=[]
try:
 with (r/'server.log').open('w') as log:
  v=subprocess.Popen(['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit=velo-q5-eager','-p','MemoryMax=116G','-p','MemorySwapMax=0','-p','KillSignal=SIGKILL','-p','WorkingDirectory='+str(q.SOURCE),'-p','Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu',*cmd],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  for _ in range(100):
   pid=a.command('systemctl','--user','show','velo-q5-eager','-p','MainPID','--value')
   if pid.isdigit() and int(pid):break
   time.sleep(.2)
  (r/'velo.pid').write_text(pid);a.wait('http://127.0.0.1:19337',process=v)
  q.advise_closed_weights(Path(model),int(pid));(r/'phase').write_text('q5-eager-correctness')
  cases=[('math','17×23의 결과를 숫자만 답해라.',lambda t:t.strip()=='391'),('korean','다음 문장만 그대로 출력해라: 고양이는 포유류입니다.',lambda t:'고양이는 포유류입니다' in t),('code','Write only a Python function add(a, b) that returns a + b.',lambda t:'def add' in t and 'return' in t)]
  for name,prompt,check in cases:
   value=a.stream('http://127.0.0.1:19337','q5-eager-audit',[{'role':'user','content':prompt}],96);value['name']=name;value['pass']=check(value['text']);results.append(value);(r/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(name,value['pass'],value['text'],flush=True)
finally:
 (r/'phase').write_text('q5-eager-cleanup');subprocess.run(['systemctl','--user','stop','velo-q5-eager'],capture_output=True)
 if v is not None:
  try:v.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(v.pid,signal.SIGKILL)
 (r/'monitor.stop').touch();mon.wait(timeout=10);(r/'phase').write_text('q5-eager-finished')
