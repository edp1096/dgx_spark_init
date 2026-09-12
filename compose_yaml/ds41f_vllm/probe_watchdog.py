"""External, opt-in probe monitor; can kill only this probe's labelled container."""
import argparse,json,os,subprocess,threading,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rank',type=int,required=True);p.add_argument('--token',required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--duration',type=int,default=3600)
p.add_argument('--min-available-gib',type=float,default=8);p.add_argument('--max-cgroup-gib',type=float,default=98)
p.add_argument('--stream-events',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
name=f'ds41-stream-{a.rank}';end=time.monotonic()+a.duration;attached=set();lock=threading.Lock();children=[]
stream=(a.output/f'memory-rank{a.rank}.jsonl').open('a',buffering=1)
def emit(row):
 row={'time_ns':time.time_ns(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),**row}
 with lock:
  stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
  if a.stream_events:print(json.dumps(row),flush=True)
def record(cmd,path):
 with path.open('ab',buffering=0) as f:
  proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  children.append(proc)
  try:
   for line in proc.stdout:
    f.write(line);os.fsync(f.fileno())
    if a.stream_events:
     with lock:print(json.dumps({'time_ns':time.time_ns(),'event':'log','source':path.name,'line':line.decode(errors='replace')}),flush=True)
    if time.monotonic()>end:break
  finally:proc.terminate()
def metadata():
 try:return json.loads(subprocess.check_output(['docker','inspect',name],stderr=subprocess.DEVNULL,timeout=2))[0]
 except (subprocess.SubprocessError,ValueError):return None
threading.Thread(target=record,args=(['journalctl','-k','-f','-n','0','-o','short-iso'],a.output/f'kernel-rank{a.rank}.log'),daemon=True).start()
emit({'event':'watchdog_started','token':a.token,'pid':os.getpid(),'min_available_gib':a.min_available_gib,'max_cgroup_gib':a.max_cgroup_gib})
(a.output/f'ready-rank{a.rank}.json').write_text(json.dumps({'pid':os.getpid(),'token':a.token,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})+'\n')
seen=False
while time.monotonic()<end and not (a.output/f'stop-rank{a.rank}').exists():
 info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=3}
 row={'available_bytes':info['MemAvailable'],'swap_free_bytes':info['SwapFree'],'memory_pressure':Path('/proc/pressure/memory').read_text().strip()}
 row['thermal_millic']={}
 for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
  try: row['thermal_millic'][(zone/'type').read_text().strip()]=int((zone/'temp').read_text())
  except (OSError,ValueError):pass
 d=metadata()
 if d and (d['Config'].get('Labels') or {}).get('ds41.probe')==a.token:
  seen=True;cid=d['Id'];row.update(container_id=cid,status=d['State']['Status'],oom_killed=d['State']['OOMKilled'])
  if cid not in attached:
   threading.Thread(target=record,args=(['docker','logs','-f','--timestamps',cid],a.output/f'container-rank{a.rank}-{cid[:12]}.log'),daemon=True).start();attached.add(cid)
  pid=d['State']['Pid']
  if pid:
   try:
    rel=next(x.split(':',2)[2] for x in Path(f'/proc/{pid}/cgroup').read_text().splitlines() if x.startswith('0::'))
    cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
    row['cgroup']={k:int((cg/('memory.'+k)).read_text()) for k in ('current','peak','max')}
    row['events']={x.split()[0]:int(x.split()[1]) for x in (cg/'memory.events').read_text().splitlines()}
   except (OSError,ValueError,StopIteration) as e:row['cgroup_error']=str(e)
  reason=None
  if d['State']['Running']:
   if info['MemAvailable']<a.min_available_gib*2**30:reason='host_available_below_floor'
   if row.get('cgroup',{}).get('current',0)>a.max_cgroup_gib*2**30:reason='container_above_guard_budget'
   if row.get('events',{}).get('oom',0):reason='container_oom_event'
  if reason:
   emit(row|{'event':'guard_trip','reason':reason})
   # Use the immutable container ID just checked; never kill a replacement.
   try:subprocess.run(['docker','kill',cid],timeout=10,check=True,capture_output=True)
   except subprocess.SubprocessError as e:emit({'event':'guard_kill_error','error':str(e)})
   (a.output/f'tripped-rank{a.rank}.json').write_text(json.dumps(row|{'reason':reason},indent=2)+'\n')
   break
 emit(row)
 if seen and d and not d['State']['Running']:break
 time.sleep(.5)
emit({'event':'watchdog_finished'})
for child in children:
 child.terminate()
 try:child.wait(timeout=2)
 except subprocess.TimeoutExpired:child.kill()
stream.close()
