import json,urllib.request,hashlib
from transformers import AutoTokenizer
from pathlib import Path
model='edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4';tok=AutoTokenizer.from_pretrained('/hf/'+model,local_files_only=True)
count=1800;lines=['Record %06d: ordinary filler text.'%i for i in range(count)];lines.insert(count//2,'The secret retrieval key is KEY_1800_0.');prompt='Read the records and return only the secret retrieval key.\n'+'\n'.join(lines)+'\nReturn the secret retrieval key only.'
messages=[{'role':'user','content':prompt}];ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
if not isinstance(ids,list):ids=ids['input_ids']
req=urllib.request.Request('http://127.0.0.1:8012/v1/tokenize',data=json.dumps({'model':model,'messages':messages,'chat_template_kwargs':{'enable_thinking':False}}).encode(),headers={'Content-Type':'application/json'})
d=json.load(urllib.request.urlopen(req));server=d['tokens'];diff=[(i,a,b) for i,(a,b) in enumerate(zip(ids,server)) if a!=b]
result={'local_count':len(ids),'server_count':len(server),'identical':ids==server,'first_differences':diff[:15],'local_tail':tok.decode(ids[-30:]),'server_tail':tok.decode(server[-30:])};print(json.dumps(result),flush=True);Path('/results/tokenization-comparison.json').write_text(json.dumps(result,indent=2))
