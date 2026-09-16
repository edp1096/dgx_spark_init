"""Host-side durable memory guard, limited to a labelled TP2 container."""
import argparse,json,subprocess,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--token',required=True);p.add_argument('--container');a=p.parse_args()
out=Path.home()/'.local/state/qwen38-tp2'/a.token;out.mkdir(parents=True,exist_ok=True)
name=a.container or f'sglang-qwen38-fn-tp2-{a.rank}'
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def emit(row):
    row=dict(time_ns=time.time_ns(),boot_id=boot,**row)
    f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
with (out/f'memory-rank{a.rank}.jsonl').open('a',buffering=1) as f:
    emit({'event':'started','pid':os.getpid(),'host_floor_gib':8,'cgroup_ceiling_gib':98})
    (out/f'ready-rank{a.rank}.json').write_text(json.dumps({'pid':os.getpid(),'boot_id':boot}))
    seen=False
    while not (out/f'stop-rank{a.rank}').exists():
        mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if len(s.split())>=3}
        row={'available_bytes':mem['MemAvailable'],'swap_free_bytes':mem['SwapFree']}
        try:
            q=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=5)
        except subprocess.TimeoutExpired:
            emit(row|{'event':'inspect_timeout'});time.sleep(.5);continue
        if q.returncode==0:
            d=json.loads(q.stdout)[0]
            if d['Config'].get('Labels',{}).get('qwen.tp2.probe')==a.token:
                seen=True;row.update(container_id=d['Id'],status=d['State']['Status'],oom=d['State']['OOMKilled'])
                pid=d['State']['Pid']
                if pid:
                    try:
                        rel=next(s.split(':',2)[2] for s in Path(f'/proc/{pid}/cgroup').read_text().splitlines() if s.startswith('0::'))
                        cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
                        row['cgroup_bytes']=int((cg/'memory.current').read_text())
                        row['memory_events']=(cg/'memory.events').read_text()
                    except (OSError,StopIteration):pass
                if d['State']['Running'] and (mem['MemAvailable']<8*2**30 or row.get('cgroup_bytes',0)>98*2**30 or row['oom']):
                    emit(row|{'event':'guard_trip'});(out/f'tripped-rank{a.rank}.json').write_text(json.dumps(row))
                    subprocess.run(['docker','kill',d['Id']],capture_output=True,timeout=15);break
                if d['State']['Status'] in ('exited','dead'):
                    emit(row);break
        emit(row)
        time.sleep(.5)
    emit({'event':'finished'})
