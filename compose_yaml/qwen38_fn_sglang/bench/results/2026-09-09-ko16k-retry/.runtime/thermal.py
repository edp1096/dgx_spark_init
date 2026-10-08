import datetime,json,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent.parent
with (out/'thermal.jsonl').open('w',buffering=1) as f:
 for _ in range(1440):
  if (out/'complete.json').exists() or (out/'error.json').exists():break
  now=datetime.datetime.now().astimezone().isoformat()
  try:
   p=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks.current.sm','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
   fields=[x.strip() for x in p.stdout.strip().split(',')]
   row=dict(time=now,values=dict(zip(['temperature_c','gpu_util_pct','power_w','sm_clock_mhz'],fields)),exit_code=p.returncode)
  except Exception as e:row=dict(time=now,error=repr(e))
  f.write(json.dumps(row)+'\n');time.sleep(10)
print('Thermal observation finished')
