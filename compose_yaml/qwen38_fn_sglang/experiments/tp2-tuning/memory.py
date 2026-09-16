import json,time,subprocess,concurrent.futures
from pathlib import Path
root=Path(__file__).parent/'results';stop=root/'stop-memory'
with (root/'memory.jsonl').open('a') as out:
 while not stop.exists():
  rows=[]
  for rank in [0,1]:
   cmd=['cat','/proc/meminfo'] if rank==0 else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','edp1096@192.168.100.60','cat /proc/meminfo']
   try:
    text=subprocess.check_output(cmd,text=True,timeout=8);vals={l.split(':')[0]:int(l.split()[1])*1024 for l in text.splitlines() if l.startswith(('MemTotal:','MemAvailable:','SwapFree:'))};rows.append({'rank':rank,**vals})
   except Exception as e:rows.append({'rank':rank,'error':str(e)})
  out.write(json.dumps({'time':time.time(),'nodes':rows})+'\n');out.flush();time.sleep(2)
