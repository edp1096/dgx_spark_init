"""Paired TG check with identical requested output and cold expert maps."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json,subprocess,time,urllib.request
from pathlib import Path
r=Path(__file__).resolve().parents[1];out=r/'results/decoder-fixed-output.json'
ssh=['ssh','-o','BatchMode=yes','edp1096@192.168.100.60']
previous=json.loads((r/'results/decoder-rows-continuation.json').read_text())
targets={x['prompt_id']:x['reply']['choices'][0]['message']['content'] for x in previous if x['kind']=='decode' and not x['enabled']}
base='\n'.join(f'Record {i}: the sample color is blue and the value is {i%23}.' for i in range(210))
rows=[]
for trial in range(2):
 for pi,target in targets.items():
  for enabled in ((False,True) if trial==0 else (True,False)):
   epoch=time.time_ns();c={'epoch':epoch,'graphs':True,'expert_io':'batch_overlap','kernel_tokens':2048,'shared_buffers':True,'final_decoder_rows':enabled,'validate_final_decoder':False}
   data=json.dumps(c).encode();p=r/'graph-control.json';tmp=p.with_suffix('.json.next');tmp.write_bytes(data);tmp.replace(p)
   subprocess.run(ssh+[f"cat > '{p}.next' && mv '{p}.next' '{p}'"],input=data,check=True)
   prompt=base+'\nIgnore the records above. Copy the text between BEGIN_COPY and END_COPY verbatim. Do not add anything or change any character. Preserve any code fences already in the text.\nBEGIN_COPY\n'+target+'\nEND_COPY\nReturn only the exact copied text.'
   req={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':prompt}],'temperature':0,'seed':42,'max_completion_tokens':256,'chat_template_kwargs':{'thinking':False},'cache_salt':f'decoder-fixed-{epoch}'}
   request=urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(request,timeout=600) as response:reply=json.load(response)
   row={'trial':trial,'prompt_id':pi,'enabled':enabled,'reply':reply,'exact_output':reply['choices'][0]['message']['content'].strip()==target.strip()}
   rows.append(row);out.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
   assert row['exact_output'] and reply['choices'][0]['finish_reason']=='stop',row
   u=reply['usage'];m=reply['metrics']
   print(json.dumps({'trial':trial,'prompt_id':pi,'enabled':enabled,'output_tokens':u['completion_tokens'],'tg':(u['completion_tokens']-1)/(m['generation_time_ms']/1000)}),flush=True)
print('FIXED_OUTPUT_TG_PASS',flush=True)
