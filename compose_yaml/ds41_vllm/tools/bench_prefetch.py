"""Alternate on/off trials with fixed policy and empty expert caches per suite."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json,subprocess,time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--suite',choices=['varied','heldout','long'],default='heldout')
p.add_argument('--repeat',type=int,default=1)
p.add_argument('--modes',nargs='+',default=['off','early','early','off'])
p.add_argument('--warmup',action='store_true')
p.add_argument('--name',default='prefetch-heldout')
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
results=root/'results'
policy=json.loads((results/'prefetch-selected-policy.json').read_text())
remote='/home/edp1096/workspace/dgx_spark_init/compose_yaml/ds41_vllm/prefetch-control.json'
def control(mode,index):
    value=policy.copy() if mode=='early' else {'mode':'off'}
    value['epoch']=time.time_ns()
    path=root/'prefetch-control.json.next'
    path.write_text(json.dumps(value))
    subprocess.run(['scp','-q','-o','BatchMode=yes',str(path),
                    'edp1096@192.168.100.60:'+remote+'.next'],check=True)
    subprocess.run(['ssh','-o','BatchMode=yes','edp1096@192.168.100.60',
                    'mv '+remote+'.next '+remote],check=True)
    path.replace(root/'prefetch-control.json')
    return value
def trial(mode,index):
    settings=control(mode,index)
    stem=f'{args.name}-{index}-{mode}'
    output=results/(stem+'.json')
    with (results/(stem+'.log')).open('w') as log:
        subprocess.run(['python3',str(root/'tools/smoke.py'),'--suite',args.suite,
                        '--repeat',str(args.repeat),'--output',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
    logs=subprocess.check_output(['docker','logs','--tail','100','ds41-stream-0'],stderr=subprocess.STDOUT).decode()
    (results/(stem+'-engine.log')).write_text(logs)
    counters=[json.loads(line.split('EXPERT_STREAM ',1)[1]) for line in logs.splitlines() if 'EXPERT_STREAM ' in line]
    rows=json.loads(output.read_text())
    info={'mode':mode,'index':index,'control':settings,'output':output.name,
          'total_seconds':sum(r['total_seconds'] for r in rows),
          'decode_tps':[r['decode_tokens_per_second'] for r in rows],
          'last_expert_counters':counters[-1] if counters else None}
    print(json.dumps(info),flush=True)
    return info
if args.warmup: trial('off','warmup')
runs=[]
for index,mode in enumerate(args.modes):
    runs.append(trial(mode,index))
    (results/(args.name+'-runs.json')).write_text(json.dumps(runs,indent=2))
reference=json.loads((results/runs[0]['output']).read_text())
for run in runs[1:]:
    current=json.loads((results/run['output']).read_text())
    if len(current)!=len(reference) or any(x['prompt']!=y['prompt'] or x['text']!=y['text'] for x,y in zip(reference,current)):
        raise SystemExit('Forecasting changed an output; inspect the recorded results')
print('ALL_TRIAL_OUTPUTS_IDENTICAL',flush=True)
