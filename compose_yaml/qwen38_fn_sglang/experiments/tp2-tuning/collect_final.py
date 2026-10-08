"""Archive selected runtime identity and memory guards, without user content."""
import json,subprocess,shlex,time,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent;out=root/'results'
code='''import json,subprocess
from pathlib import Path
d=json.loads(subprocess.check_output(['docker','inspect',NAME]))[0]
token=d['Config']['Labels'].get('qwen.tp2.probe')
p=Path.home()/'.local/state/qwen38-tp2'/token
memory=p/MEMORY
rows=[json.loads(s) for s in memory.read_text().splitlines()] if memory.exists() else []
print(json.dumps({'rank':RANK,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'token':token,'image_id':d['Image'],'image':d['Config']['Image'],'state':d['State'],'watchdog':(p/WATCHDOG).read_text() if (p/WATCHDOG).exists() else None,'memory_first':rows[0] if rows else None,'memory_last':rows[-1] if rows else None,'memory_samples':len(rows)}))
'''
rows=[]
for rank in (0,1):
 script=code.replace('NAME',repr(f'sglang-qwen38-fn-tp2-{rank}')).replace('MEMORY',repr(f'memory-rank{rank}.jsonl')).replace('WATCHDOG',repr(f'watchdog-{rank}.log')).replace('RANK',str(rank))
 cmd=['python3','-c',script]
 if rank:cmd=['ssh','-o','BatchMode=yes','edp1096@192.168.100.60',shlex.join(cmd)]
 rows.append(json.loads(subprocess.check_output(cmd,text=True)))
 info=subprocess.check_output(([] if rank==0 else ['ssh','-o','BatchMode=yes','edp1096@192.168.100.60'])+['docker','logs',f'sglang-qwen38-fn-tp2-{rank}'],stderr=subprocess.STDOUT)
 (out/f'final-rank{rank}.log').write_bytes(info)
d={'time':time.time(),'nodes':rows,'server':json.load(urllib.request.urlopen('http://127.0.0.1:8012/get_server_info'))}
(out/'final-runtime.json').write_text(json.dumps(d,indent=2)+'\n')
subprocess.run(['python3',str(root/'capture_state.py')],check=True)
print([(r['rank'],r['state']['Status'],r['state']['OOMKilled'],r['watchdog']) for r in rows])
