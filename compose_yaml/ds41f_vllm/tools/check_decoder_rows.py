"""Untimed full-layer equivalence and cached-prefix continuation checks."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json,subprocess,time,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixtures',type=Path,required=True)
p.add_argument('--output',type=Path,default=root/'results/decoder-rows-correctness.json')
a=p.parse_args()
ssh=['ssh','-o','BatchMode=yes','edp1096@192.168.100.60']

def control(enabled,validate=False):
    epoch=time.time_ns()
    obj={'epoch':epoch,'graphs':True,'expert_io':'batch_overlap','kernel_tokens':2048,
         'shared_buffers':True,'final_decoder_rows':enabled,'validate_final_decoder':validate}
    data=json.dumps(obj).encode()
    path=root/'graph-control.json';tmp=path.with_suffix('.json.next');tmp.write_bytes(data);tmp.replace(path)
    subprocess.run(ssh+[f"cat > '{path}.next' && mv '{path}.next' '{path}'"],input=data,check=True)
    return epoch

def call(request):
    req=urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as r:return json.load(r)

for _ in range(600):
    try:urllib.request.urlopen('http://127.0.0.1:8010/health',timeout=2).close();break
    except OSError:time.sleep(1)
else:raise RuntimeError('health timeout')
fixtures=json.loads(a.fixtures.read_text())
results=[]
for fixture in fixtures:
    epoch=control(True,True)
    request=dict(fixture['request'])
    request.update(stream=False,max_completion_tokens=16,temperature=0,chat_template_kwargs={'thinking':False},cache_salt=f'decoder-validation-{epoch}')
    request.pop('stream_options',None);request.pop('reasoning_effort',None)
    reply=call(request)
    row={'fixture':fixture['name'],'reply':reply,'ranks':{}}
    # Compare full and selected final-layer states in the actual model, on
    # both ranks and on every prefill chunk. No validation timing is a benchmark.
    for rank in (0,1):
        cmd=['docker','logs','--since','20m',f'ds41-stream-{rank}']
        log=subprocess.check_output(cmd if rank==0 else ssh+cmd,stderr=subprocess.STDOUT).decode()
        log=log.split('EXPERT_CACHE_RESET '+str(epoch),1)[1]
        (root/'results'/f'decoder-validation-{fixture["name"]}-rank{rank}.log').write_text(log)
        checks=[json.loads(x.split('FINAL_DECODER_ROWS ',1)[1]) for x in log.splitlines() if 'FINAL_DECODER_ROWS {' in x]
        assert checks and all('validation' in x for x in checks), (rank,checks)
        row['ranks'][str(rank)]=checks
        assert all(v['within_tolerance'] for c in checks for v in c['validation']),checks
    assert reply['choices'][0]['message']['content'].strip().strip('`". *')==fixture['expected'],reply
    assert reply['choices'][0]['finish_reason']=='stop',reply
    results.append(row);a.output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('VALIDATED '+fixture['name'],flush=True)
print('DECODER_ROWS_FULL_LAYER_VALIDATION_PASS',flush=True)
