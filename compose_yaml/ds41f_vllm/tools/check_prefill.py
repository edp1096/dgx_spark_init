"""Exercise more than one 512-token prefill chunk and verify a text marker."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output',default=str((Path(__file__).resolve().parents[1] / 'results')/'b12x-prefill-marker.json'))
args=p.parse_args()
lines=[f'Record {i:03d}: sample item {i} has reference number {10000+i} and belongs to the blue group.' for i in range(50)]
lines.insert(23,'Validation marker: spark-cobalt-731')
prompt='Read the records and return only the exact validation marker.\n'+'\n'.join(lines)
body={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':prompt}],
      'max_tokens':16,'temperature':0,'stream':True,'stream_options':{'include_usage':True},
      'chat_template_kwargs':{'thinking':False},'return_token_ids':True}
request=urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
start=time.monotonic();first=None;text='';usage=None
with urllib.request.urlopen(request,timeout=600) as response:
    for line in response:
        if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]': continue
        event=json.loads(line[6:])
        if event.get('usage'): usage=event['usage']
        for choice in event.get('choices',[]):
            part=choice.get('delta',{}).get('content')
            if part:
                if first is None: first=time.monotonic()
                text+=part
result={'text':text,'usage':usage,'ttft_seconds':first-start if first else None,'total_seconds':time.monotonic()-start}
Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False),flush=True)
assert usage and usage['prompt_tokens']>512
assert text.strip().strip('`"').strip()=='spark-cobalt-731'
print('PREFILL_MARKER_PASS',flush=True)
