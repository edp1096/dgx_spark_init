"""Diagnose a greedy continuation mismatch against repeated full-path controls."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json,subprocess,time,urllib.request
from pathlib import Path
r=Path(__file__).resolve().parents[1];out=r/'results/decoder-repeat.json'
ssh=['ssh','-o','BatchMode=yes','edp1096@192.168.100.60']
base='\n'.join(f'Record {i}: the sample color is blue and the value is {i%23}.' for i in range(210))
prompt=base+'\nIgnore the sample records for this task. '+'Write a Python function implementing binary search on a sorted list of integers. Include type hints and a concise docstring. Return only code.'
rows=[]
for enabled in (False,False,True,True):
 epoch=time.time_ns()
 c={'epoch':epoch,'graphs':True,'expert_io':'batch_overlap','kernel_tokens':2048,'shared_buffers':True,'final_decoder_rows':enabled,'validate_final_decoder':enabled}
 data=json.dumps(c).encode();p=r/'graph-control.json';tmp=p.with_suffix('.json.next');tmp.write_bytes(data);tmp.replace(p)
 subprocess.run(ssh+[f"cat > '{p}.next' && mv '{p}.next' '{p}'"],input=data,check=True)
 req={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':prompt}],'temperature':0,'seed':42,'max_completion_tokens':180,'chat_template_kwargs':{'thinking':False},'cache_salt':f'decoder-repeat-{epoch}'}
 request=urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=600) as response:reply=json.load(response)
 row={'enabled':enabled,'reply':reply,'ranks':{}}
 for rank in (0,1):
  cmd=['docker','logs','--since','10m',f'ds41-stream-{rank}']
  log=subprocess.check_output(cmd if rank==0 else ssh+cmd,stderr=subprocess.STDOUT).decode().split('EXPERT_CACHE_RESET '+str(epoch),1)[1]
  (r/'results'/f'decoder-repeat-{len(rows)}-rank{rank}.log').write_text(log)
  row['ranks'][str(rank)]=[json.loads(s.split('FINAL_DECODER_ROWS ',1)[1]) for s in log.splitlines() if 'FINAL_DECODER_ROWS {' in s]
 rows.append(row);out.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'enabled':enabled,'text':reply['choices'][0]['message']['content'],'checks':row['ranks']},ensure_ascii=False),flush=True)
