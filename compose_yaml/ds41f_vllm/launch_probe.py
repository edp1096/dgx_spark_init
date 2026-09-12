"""Launch a labelled two-node probe with independent, durable monitoring."""
import argparse,json,os,re,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--token',required=True);p.add_argument('--batch',type=int,choices=(2048,4096),required=True)
p.add_argument('--bench-control',action='store_true')
a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9_-]+',a.token)
root=Path(__file__).resolve().parent
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','edp1096@192.168.100.60']
for rank in (0,1):
 cmd=['docker','inspect',f'ds41-stream-{rank}']
 result=subprocess.run(cmd if rank==0 else ssh+[shlex.join(cmd)],capture_output=True,text=True,timeout=10)
 if result.returncode==0:assert not json.loads(result.stdout)[0]['State']['Running'],'Stop both ranks before launching a probe'
out=Path.home()/'.local/state/ds41-probes'/a.token
out.mkdir(parents=True,exist_ok=True)
ready=out/'ready-rank0.json'
if ready.exists():
 x=json.loads(ready.read_text())
 if Path(f"/proc/{x['pid']}").exists():raise RuntimeError('This probe token already has a watcher; use a fresh token')
 ready.unlink()
code=f'''from pathlib import Path
import json,subprocess
root=Path({str(root)!r});token={a.token!r}
out=Path.home()/'.local/state/ds41-probes'/token
out.mkdir(parents=True,exist_ok=True)
processes={{}}
for script,args in [('probe_watchdog.py',['--rank','1']),('probe_peer_recorder.py',[])]:
 with (out/(script+'.log')).open('ab') as f:
  child=subprocess.Popen(['python3',str(root/script),*args,'--token',token,'--output',str(out)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 processes[script]=child.pid
(out/'observer-processes.json').write_text(json.dumps(processes)+'\\n')
print(json.dumps(processes))
'''
subprocess.run(ssh+['python3','-'],input=code,text=True,check=True)
for _ in range(75):
 if ready.exists():break
 time.sleep(.2)
else:raise RuntimeError('Head watchdog did not become ready')
x=json.loads(ready.read_text());os.kill(x['pid'],0)
assert x['token']==a.token and x['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
env=os.environ|{'DSV41_PROBE_TOKEN':a.token,'DSV41_MAX_BATCHED_TOKENS':str(a.batch),'DSV41_BENCH_CONTROL':'1' if a.bench_control else '0'}
subprocess.run(['./manage.sh','start'],cwd=root,env=env,check=True)
print('PROBE_STARTED '+a.token,flush=True)
