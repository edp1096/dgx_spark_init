"""Exact-length, three-position retrieval probe. Run in the serving image."""
import argparse,json,random,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--context',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--url',default='http://127.0.0.1:8013');a=p.parse_args()
from transformers import AutoTokenizer
MODEL='/hf/hub/models--dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4/snapshots/be794b990578ef3031eccf9f28e675a289a09ee9'
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
rng=random.Random(941)
expected={k:''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789',k=12)) for k in ('amber','birch','cobalt')}
enc=lambda s:tok.encode(s,add_special_tokens=False)
prefix=enc('<|im_start|>system\nRead the supplied archive and return the requested exact registry values. Answer only with a JSON object. Do not reason aloud.<|im_end|>\n<|im_start|>user\nArchive begins.\n')
footer=enc('\nArchive ends. Return the SECRET_REGISTRY values for amber, birch, and cobalt as JSON.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
needles=[enc(f'\nSECRET_REGISTRY {k} = {v}\n') for k,v in expected.items()]
target=a.context-2048
filler=[]
for i in range(4096):
 text=f'Archive record {i}: zone {rng.randrange(10000)}; reading {rng.randrange(1000000)}; status '+rng.choice(['inspected','stored','released','pending'])+'; description '+rng.choice(['a blue ceramic cup','a wooden shipping box','a green metal lamp','a linen document folder'])+'.\n'
 filler.extend(enc(text))
ids=list(prefix);positions=[]
for j,needle in enumerate(needles):
 boundary=int(target*(.1,.5,.9)[j]);n=boundary-len(ids);ids.extend((filler*((n//len(filler))+1))[:n]);positions.append(len(ids));ids.extend(needle)
n=target-len(ids)-len(footer);assert n>=0
ids.extend((filler*((n//len(filler))+1))[:n]);ids.extend(footer);assert len(ids)==target
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
base={'context':a.context,'input_tokens':len(ids),'needle_positions':positions,'expected':expected,'kv_dtype':'fp8_e4m3','time_started':time.time()}
out.write_text(json.dumps(base,indent=2)+'\n')
request={'model':'qwen38-b12x-tp2','prompt':ids,'temperature':0,'max_tokens':256,'stream':True,'stream_options':{'include_usage':True}}
start=time.monotonic();first=None;last=None;text='';meta={}
try:
 req=urllib.request.Request(a.url+'/v1/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=7200) as response:
  for line in response:
   if not line.startswith(b'data: '):continue
   raw=line[6:].strip()
   if raw==b'[DONE]':break
   row=json.loads(raw);last=row
   if row.get('usage'):meta=row['usage']
   delta=''.join(c.get('text','') for c in row.get('choices',[]))
   text+=delta
   if delta and first is None:first=time.monotonic()-start
 elapsed=time.monotonic()-start
 assert last is not None,'No response'
 last={'text':text,'usage':meta}
 assert meta.get('prompt_tokens')==len(ids),f'Prompt count mismatch: {meta}'
 try:
  left=text.index('{');right=text.rindex('}')+1;actual=json.loads(text[left:right])
 except (ValueError,json.JSONDecodeError):actual={}
 matched={k:actual.get(k)==v for k,v in expected.items()}
 base.update(elapsed_s=elapsed,ttft_s=first,tg_estimate=(meta.get('completion_tokens',1)-1)/(elapsed-first) if first and elapsed>first else None,pp_estimate=len(ids)/first if first else None,response=last,matched=matched,passed=all(matched.values()))
except Exception as e:
 base.update(elapsed_s=time.monotonic()-start,error=repr(e),passed=False)
out.write_text(json.dumps(base,indent=2,ensure_ascii=False)+'\n');print(json.dumps(base,ensure_ascii=False),flush=True)
raise SystemExit(0 if base['passed'] else 1)
