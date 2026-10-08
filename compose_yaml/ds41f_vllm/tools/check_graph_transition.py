"""Compare eager and graph outputs while decode crosses the 512-token bound."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json,subprocess,time,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base='http://127.0.0.1:8010'
model='deepseek-v4.1-flash'

def post(path,body):
    request=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=300) as response: return json.load(response)

def prompt(n):
    return ('Ignore this padding block:\n'+'pad '*n+'\nWrite a Python function implementing binary search on a sorted list of integers. Include type hints and a concise docstring. Return only code.')

def messages(n): return [{'role':'user','content':prompt(n)}]

lo,hi=0,600
while lo<hi:
    mid=(lo+hi+1)//2
    count=post('/tokenize',{'model':model,'messages':messages(mid),'add_generation_prompt':True,'chat_template_kwargs':{'thinking':False}})['count']
    if count<=500:lo=mid
    else:hi=mid-1
rows=[]
for enabled in (False,True):
    control=root/'graph-control.json.next'
    control.write_text(json.dumps({'epoch':time.time_ns(),'graphs':enabled}))
    remote='/home/edp1096/workspace/dgx_spark_init/compose_yaml/ds41f_vllm/graph-control.json'
    subprocess.run(['scp','-q','-o','BatchMode=yes',str(control),'edp1096@192.168.100.60:'+remote+'.next'],check=True)
    subprocess.run(['ssh','-o','BatchMode=yes','edp1096@192.168.100.60','mv '+remote+'.next '+remote],check=True)
    control.replace(root/'graph-control.json')
    start=time.monotonic()
    result=post('/v1/chat/completions',{'model':model,'messages':messages(lo),'temperature':0,'max_tokens':180,'chat_template_kwargs':{'thinking':False}})
    row={'graphs':enabled,'seconds':time.monotonic()-start,'usage':result['usage'],'text':result['choices'][0]['message']['content'],'finish_reason':result['choices'][0]['finish_reason']}
    assert 480<=row['usage']['prompt_tokens']<=506,row
    assert row['usage']['total_tokens']>530,row
    assert row['finish_reason']=='stop',row
    rows.append(row);print(json.dumps(row),flush=True)
    (root/'results'/'graph-context-transition.json').write_text(json.dumps(rows,indent=2))
assert rows[0]['text']==rows[1]['text'] and rows[0]['usage']==rows[1]['usage']
print('CONTEXT_GRAPH_TRANSITION_PASS',flush=True)
