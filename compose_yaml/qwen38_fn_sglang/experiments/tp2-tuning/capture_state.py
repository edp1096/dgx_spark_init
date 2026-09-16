import json,subprocess,shlex,time
from pathlib import Path
r=Path(__file__).resolve().parent
code='''import json,subprocess
from pathlib import Path
d=json.loads(subprocess.check_output(['docker','inspect',NAME]))[0]
x={'rank':RANK,'token':d['Config']['Labels'].get('qwen.tp2.probe'),'image':d['Image'],'state':d['State']}
pid=d['State']['Pid']
if pid:
 rel=next(s.split(':',2)[2] for s in Path(f'/proc/{pid}/cgroup').read_text().splitlines() if s.startswith('0::'))
 cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
 for name in ['memory.current','memory.peak','memory.events']:
  p=cg/name
  if p.exists():x[name]=p.read_text().strip()
print(json.dumps(x))
'''
rows=[]
for rank in [0,1]:
 script=code.replace('NAME',repr(f'sglang-qwen38-fn-tp2-{rank}')).replace('RANK',str(rank))
 cmd=['python3','-c',script]
 if rank:cmd=['ssh','-o','BatchMode=yes','edp1096@192.168.100.60',shlex.join(cmd)]
 try:rows.append(json.loads(subprocess.check_output(cmd,text=True)))
 except Exception as e:rows.append({'rank':rank,'error':str(e)})
with (r/'results/container-peaks.jsonl').open('a') as out:out.write(json.dumps({'time':time.time(),'nodes':rows})+'\n')
print([(x.get('rank'),x.get('token'),x.get('memory.peak')) for x in rows])
