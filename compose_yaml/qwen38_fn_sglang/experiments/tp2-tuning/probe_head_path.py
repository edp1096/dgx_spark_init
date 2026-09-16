import json,time,urllib.request,urllib.error
from pathlib import Path
from transformers import AutoTokenizer
model='/hf/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4'
tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
count=1800;key='KEY_1800_0';lines=['Record %06d: ordinary filler text.'%i for i in range(count)];lines.insert(count//2,'The secret retrieval key is '+key+'.')
prompt='Read the records and return only the secret retrieval key.\n'+'\n'.join(lines)+'\nReturn the secret retrieval key only.'
ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
if not isinstance(ids,list):ids=ids['input_ids']
rows=[];url='http://127.0.0.1:8012'
for repeat in range(4):
 for logprobs in [False,True]:
  for _ in range(40):
   try:urllib.request.urlopen(url+'/flush_cache',timeout=10).read();break
   except urllib.error.HTTPError:time.sleep(.25)
  body={'input_ids':ids,'sampling_params':{'temperature':0,'top_p':.95,'top_k':20,'max_new_tokens':32},'stream':False,'return_logprob':logprobs}
  if logprobs:body['logprob_start_len']=len(ids)-3
  req=urllib.request.Request(url+'/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  try:
   result=json.load(urllib.request.urlopen(req,timeout=180));text=result['text'];row={'repeat':repeat,'logprobs':logprobs,'tokens':len(ids),'text':text,'passed':text.strip()==key}
  except urllib.error.HTTPError as e:row={'repeat':repeat,'logprobs':logprobs,'error':e.read().decode()}
  rows.append(row);Path('/results/head-path-probe.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
