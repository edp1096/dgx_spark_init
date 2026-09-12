"""Run the head probe watcher over SSH; fsync its stream on the surviving peer."""
import argparse,json,os,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--token',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parent
cmd=['python3',str(root/'probe_watchdog.py'),'--rank','0','--token',a.token,'--output',str(a.output),'--stream-events']
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ServerAliveInterval=2','-o','ServerAliveCountMax=3','edp1096@192.168.100.61',shlex.join(cmd)]
proc=subprocess.Popen(ssh,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
with (a.output/'head-mirror-on-worker.jsonl').open('ab',buffering=0) as f:
 for line in proc.stdout:f.write(line);os.fsync(f.fileno())
code=proc.wait()
(a.output/'head-recorder-exit.json').write_text(json.dumps({'ssh_exit':code})+'\n')
if code:
 try:
  d=json.loads(subprocess.check_output(['docker','inspect','ds41-stream-1'],timeout=3))[0]
  if (d['Config'].get('Labels') or {}).get('ds41.probe')==a.token and d['State']['Running']:
   subprocess.run(['docker','kill',d['Id']],timeout=10,check=True)
 except (subprocess.SubprocessError,ValueError):pass
