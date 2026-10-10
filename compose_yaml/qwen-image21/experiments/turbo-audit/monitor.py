import json,os,signal,subprocess,sys,threading,time
from pathlib import Path
R=Path(sys.argv[1]);done=threading.Event()
def state():
 try:
  d=json.loads((R/'current.json').read_text());w=R/d['profile']/'worker-state.json'
  if w.exists():d['worker']=json.loads(w.read_text())
  return d
 except (FileNotFoundError,json.JSONDecodeError):return {}
def gpu():
 with (R/'gpu.jsonl').open('a',buffering=1) as f:
  while not done.is_set():
   p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True);f.write(json.dumps({'wall':time.time(),'state':state(),'processes':p.stdout.strip()})+'\n');done.wait(1)
t=threading.Thread(target=gpu);t.start();group=None;last=0;resident_groups={}
try:
 with (R/'memory.jsonl').open('a',buffering=1) as f:
  while not (R/'monitor.stop').exists():
   if time.monotonic()-last>2:
    p=subprocess.run(['docker','inspect','qwim21-turbo-audit'],capture_output=True,text=True)
    if p.returncode==0:
     pid=json.loads(p.stdout)[0]['State']['Pid']
     if pid:
      try:group=Path('/sys/fs/cgroup')/Path(f'/proc/{pid}/cgroup').read_text().split('0::')[1].strip().lstrip('/')
      except FileNotFoundError:pass
    last=time.monotonic()
    q=subprocess.run(['docker','inspect','sparktalk-qwen38fn_exl3','sparktalk-nemotron-asr','sparktalk-qwen3-tts','sparktalk-embedding'],capture_output=True,text=True)
    if q.returncode==0:
     resident_groups={}
     for item in json.loads(q.stdout):
      pid=item['State']['Pid']
      if pid:
       try:resident_groups[item['Name'].lstrip('/')]=Path('/sys/fs/cgroup')/Path(f'/proc/{pid}/cgroup').read_text().split('0::')[1].strip().lstrip('/')
       except FileNotFoundError:pass
   m={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if len(s.split())>=3};v=dict(s.split() for s in Path('/proc/vmstat').read_text().splitlines());row={'wall':time.time(),'state':state(),'available':m['MemAvailable'],'free':m['MemFree'],'swap':m['SwapTotal']-m['SwapFree'],'pswpout':int(v['pswpout'])}
   if group:
    try:row['cgroup']={'current':int((group/'memory.current').read_text()),'swap':int((group/'memory.swap.current').read_text()),'events':dict((k,int(v)) for k,v in (s.split() for s in (group/'memory.events').read_text().splitlines()))}
    except FileNotFoundError:pass
   row['resident_cgroups']={}
   for name,path in resident_groups.items():
    try:row['resident_cgroups'][name]={'swap':int((path/'memory.swap.current').read_text()),'events':dict((k,int(v)) for k,v in (s.split() for s in (path/'memory.events').read_text().splitlines()))}
    except FileNotFoundError:pass
   f.write(json.dumps(row)+'\n')
   if m['MemAvailable']<int(1.5*2**30):
    (R/'memory-floor.json').write_text(json.dumps(row));subprocess.run(['docker','stop','-t','3','qwim21-turbo-audit'],stdout=subprocess.DEVNULL);break
   done.wait(.25)
finally:done.set();t.join()
