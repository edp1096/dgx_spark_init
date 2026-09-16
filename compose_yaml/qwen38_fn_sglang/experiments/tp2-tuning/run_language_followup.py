from pathlib import Path
import json,time,subprocess
r=Path(__file__).resolve().parent;p=r/'results/ko64k-1024-prefill.json'
while True:
 try:
  if json.loads(p.read_text()).get('completed'):break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(2)
for file in ['language_thinking.py','language_controls.py']:
 subprocess.run(['python3',str(r/file)],check=True)
print('LANGUAGE_FOLLOWUP_COMPLETE',flush=True)
