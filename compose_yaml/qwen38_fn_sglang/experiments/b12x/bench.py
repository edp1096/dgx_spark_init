"""Same exact-token requests, warmed kernels, flushed prefix cache per trial."""
import argparse,json,time,urllib.request,statistics
from pathlib import Path
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--url',default='http://127.0.0.1:8014');a=p.parse_args()
model='/hf/hub/models--dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4/snapshots/be794b990578ef3031eccf9f28e675a289a09ee9'
tok=AutoTokenizer.from_pretrained(model,local_files_only=True);enc=lambda s:tok.encode(s,add_special_tokens=False)
def request(path,data=None):
 req=urllib.request.Request(a.url+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 return urllib.request.urlopen(req,timeout=600)
def generate(ids,maximum=128):
 start=time.monotonic();first=None;text='';meta={}
 with request('/generate',{'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':maximum},'stream':True}) as f:
  for line in f:
   if not line.startswith(b'data: '):continue
   raw=line[6:].strip()
   if raw==b'[DONE]':break
   row=json.loads(raw);text=row.get('text',text);meta=row.get('meta_info',meta)
   if text and first is None:first=time.monotonic()-start
 elapsed=time.monotonic()-start
 assert meta.get('prompt_tokens')==len(ids),meta
 return {'input_tokens':len(ids),'completion_tokens':meta.get('completion_tokens'),'cached_tokens':meta.get('cached_tokens'),'ttft_s':first,'elapsed_s':elapsed,'pp':len(ids)/first,'tg':(meta['completion_tokens']-1)/(elapsed-first),'text':text}
prefix=enc('<|im_start|>system\nFollow the user instructions.<|im_end|>\n<|im_start|>user\n')
suffix=enc('\nNow ignore the archive above. Start with the answer to 17+25, then write the integers from 1 through 100 separated by commas.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
filler=enc('Archive entry: the blue cup is stored in the wooden box. ')
rows=[]
for count in (512,4096,16384):
 n=count-len(prefix)-len(suffix);ids=prefix+(filler*((n//len(filler))+1))[:n]+suffix
 warm=generate(ids)
 for repeat in range(10 if count==4096 else 3):
  with request('/flush_cache') as f:f.read()
  row=generate(ids);row['repeat']=repeat;row['quality_ok']=row['text'].lstrip().startswith('42')
  rows.append(row);Path(a.output).write_text(json.dumps({'rows':rows},indent=2));assert row['quality_ok'],row
  print(json.dumps({k:v for k,v in row.items() if k!='text'}),flush=True)
  Path(a.output).write_text(json.dumps({'rows':rows},indent=2))
summary={str(n):{k:statistics.median(r[k] for r in rows if r['input_tokens']==n) for k in ('pp','tg','ttft_s')} for n in (512,4096,16384)}
Path(a.output).write_text(json.dumps({'rows':rows,'median':summary},indent=2));print(json.dumps(summary),flush=True)
