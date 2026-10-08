"""Controlled expert-cache resets and GPU telemetry for whole-model trials."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json,subprocess,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--label',required=True)
p.add_argument('--suite',choices=['varied','heldout','long'],default='varied')
p.add_argument('--repeat',type=int,default=1)
p.add_argument('--trials',type=int,default=2)
p.add_argument('--bypass-graphs',action='store_true')
p.add_argument('--io-modes',help='Comma-separated timed I/O modes; overrides --trials')
p.add_argument('--trace-warmup',action='store_true')
args=p.parse_args()
io_modes=args.io_modes.split(',') if args.io_modes else [None]*args.trials
if any(m not in (None,'serial','overlap','batch','batch_overlap') for m in io_modes):
    p.error('Invalid I/O mode')
root=Path(__file__).resolve().parents[1];results=root/'results'
remote='/home/edp1096/workspace/dgx_spark_init/compose_yaml/ds41f_vllm/graph-control.json'
for _ in range(360):
    try:
        urllib.request.urlopen('http://127.0.0.1:8010/health',timeout=2)
        break
    except OSError: time.sleep(1)
else: raise SystemExit('API not ready')
query=['timeout','600s','nvidia-smi','--query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu,clocks_event_reasons.active','--format=csv','-lms','200']
monitors=[];files=[]
for rank in (0,1):
    f=(results/f'{args.label}-gpu-rank{rank}.csv').open('w');files.append(f)
    cmd=query if rank==0 else ['ssh','-o','BatchMode=yes','edp1096@192.168.100.60',' '.join(query)]
    monitors.append(subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT))
try:
    runs=[]
    for trial in range(-1,len(io_modes)):
        since=datetime.now(timezone.utc).isoformat()
        control=root/'graph-control.json.next'
        settings={'epoch':time.time_ns(),'graphs':not args.bypass_graphs}
        io_mode=io_modes[max(0,trial)]
        if io_mode is not None: settings['expert_io']=io_mode
        settings['record_routes']=bool(args.trace_warmup and trial==-1)
        control.write_text(json.dumps(settings))
        subprocess.run(['scp','-q','-o','BatchMode=yes',str(control),'edp1096@192.168.100.60:'+remote+'.next'],check=True)
        subprocess.run(['ssh','-o','BatchMode=yes','edp1096@192.168.100.60','mv '+remote+'.next '+remote],check=True)
        control.replace(root/'graph-control.json')
        name=f'{args.label}-'+('warmup' if trial==-1 else str(trial))
        output=results/(name+'.json')
        with (results/(name+'.log')).open('w') as f:
            subprocess.run(['python3',str(root/'tools/smoke.py'),'--suite',args.suite,'--repeat',str(args.repeat),
                            '--output',str(output)],stdout=f,stderr=subprocess.STDOUT,check=True)
        log_cmd=['docker','logs','--since',since,'ds41-stream-0']
        logs=subprocess.check_output(log_cmd,stderr=subprocess.STDOUT).decode()
        worker_logs=subprocess.check_output(['ssh','-o','BatchMode=yes','edp1096@192.168.100.60',
            'docker','logs','--since',since,'ds41-stream-1'],stderr=subprocess.STDOUT).decode()
        (results/(name+'-engine-rank1.log')).write_text(worker_logs)
        (results/(name+'-engine.log')).write_text(logs)
        if settings['record_routes']:
            for rank in (0,1):
                source=f'/home/edp1096/.cache/ds41-stream/route-trace-{settings["epoch"]}-rank{rank}.jsonl'
                destination=results/(name+f'-routes-rank{rank}.jsonl')
                if rank:
                    subprocess.run(['scp','-q','-o','BatchMode=yes','edp1096@192.168.100.60:'+source,str(destination)],check=True)
                else:
                    import shutil
                    shutil.copyfile(source,destination)
        counters=[json.loads(line.split('EXPERT_STREAM ',1)[1]) for line in logs.splitlines() if 'EXPERT_STREAM ' in line]
        rows=json.loads(output.read_text())
        info={'trial':trial,'expert_io':io_mode,'output':output.name,'total_seconds':sum(r['total_seconds'] for r in rows),
              'decode_tps':[r['decode_tokens_per_second'] for r in rows],'last_counters':counters[-1] if counters else None}
        print(json.dumps(info),flush=True)
        if trial>=0:
            runs.append(info)
            (results/(args.label+'-runs.json')).write_text(json.dumps(runs,indent=2))
finally:
    for process in monitors: process.terminate()
    for process in monitors:
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill();process.wait()
    for f in files: f.close()
