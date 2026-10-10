"""Unified memory, swap and CUDA allocation telemetry for the isolated Q5 trial."""
import json,os,signal,subprocess,sys,threading,time
from pathlib import Path
R=Path(sys.argv[1]);model_label=sys.argv[2] if len(sys.argv)>2 else 'velo';done=threading.Event()
N=['sparktalk-nemotron-asr','sparktalk-embedding','sparktalk-extra-media']
def phase():
 try:return (R/'phase').read_text().strip()
 except FileNotFoundError:return 'startup'
def gpu():
 with (R/'gpu.jsonl').open('a',buffering=1) as f:
  while not done.is_set():
   p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
   f.write(json.dumps({'wall':time.time(),'phase':phase(),'processes':p.stdout.strip()})+'\n');done.wait(1)
t=threading.Thread(target=gpu);t.start();groups={};last=0
try:
 with (R/'memory.jsonl').open('a',buffering=1) as f:
  while not (R/'monitor.stop').exists():
   if time.monotonic()-last>2:
    ds=subprocess.run(['docker','inspect',*N],capture_output=True,text=True)
    if ds.returncode==0:
     for d in json.loads(ds.stdout):
      pid=d['State']['Pid']
      if pid:
       try:groups[d['Name'].lstrip('/')]=Path('/sys/fs/cgroup')/Path(f'/proc/{pid}/cgroup').read_text().split('0::')[1].strip().lstrip('/')
       except FileNotFoundError:pass
    last=time.monotonic()
   m={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if len(s.split())>=3}
   v=dict(s.split() for s in Path('/proc/vmstat').read_text().splitlines())
   row={'wall':time.time(),'phase':phase(),'available':m['MemAvailable'],'free':m['MemFree'],'swap':m['SwapTotal']-m['SwapFree'],'pswpin':int(v['pswpin']),'pswpout':int(v['pswpout']),'cgroups':{}}
   try:
    pid=int((R/'velo.pid').read_text());state=Path(f'/proc/{pid}/status').read_text()
    groups[model_label]=Path('/sys/fs/cgroup')/Path(f'/proc/{pid}/cgroup').read_text().split('0::')[1].strip().lstrip('/')
    row['velo']={s.split(':')[0]:int(s.split()[1])*1024 for s in state.splitlines() if s.startswith(('VmRSS:','VmHWM:','VmSwap:'))}
   except (FileNotFoundError,ValueError):pass
   for name,c in groups.copy().items():
    try:row['cgroups'][name]={'current':int((c/'memory.current').read_text()),'swap':int((c/'memory.swap.current').read_text()),'events':dict((k,int(v)) for k,v in (s.split() for s in (c/'memory.events').read_text().splitlines()))}
    except FileNotFoundError:pass
   f.write(json.dumps(row)+'\n')
   if m['MemAvailable']<int(1.5*2**30):
    (R/'memory-floor.json').write_text(json.dumps(row))
    try:os.kill(int((R/'velo.pid').read_text()),signal.SIGKILL)
    except (FileNotFoundError,ValueError,ProcessLookupError):pass
    subprocess.run(['docker','stop','--time','5',*N],stdout=subprocess.DEVNULL);break
   done.wait(.25)
finally:done.set();t.join()
