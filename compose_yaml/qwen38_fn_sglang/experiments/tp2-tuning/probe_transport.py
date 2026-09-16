import json,urllib.request
from pathlib import Path
from transformers import AutoTokenizer
from measure import flush_cache,URL,MODEL
path='/hf/'+MODEL;tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
count=1800;key='KEY_1800_0';lines=['Record %06d: ordinary filler text.'%i for i in range(count)];lines.insert(count//2,'The secret retrieval key is '+key+'.')
prompt='Read the records and return only the secret retrieval key.\n'+'\n'.join(lines)+'\nReturn the secret retrieval key only.'
messages=[{'role':'user','content':prompt}];ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
if not isinstance(ids,list):ids=ids['input_ids']
rows=[]
for repeat in range(2):
 for api,stream,seed in [('native',True,42),('native',True,None),('chat',True,42),('chat',True,None),('chat',False,42),('native',False,42)]:
  flush_cache()
  if api=='native':body={'input_ids':ids,'stream':stream,'sampling_params':{'temperature':0,'top_p':.95,'top_k':20,'max_new_tokens':32,'sampling_seed':seed}};endpoint='/generate'
  else:body={'model':MODEL,'messages':messages,'stream':stream,'temperature':0,'top_p':.95,'top_k':20,'max_tokens':32,'seed':seed,'presence_penalty':0,'chat_template_kwargs':{'enable_thinking':False}};endpoint='/v1/chat/completions'
  req=urllib.request.Request(URL+endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'});text='';revisions=[]
  with urllib.request.urlopen(req,timeout=180) as response:
   if stream:
    for line in response:
     if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
     d=json.loads(line[6:])
     if api=='native':
      new=d.get('text','')
      if not new.startswith(text):revisions.append([text,new])
      text=new
     else:
      for c in d.get('choices',[]):text+=c.get('delta',{}).get('content') or ''
   else:
    d=json.load(response);text=d['text'] if api=='native' else d['choices'][0]['message'].get('content','')
  row=dict(repeat=repeat,api=api,stream=stream,seed=seed,text=text,passed=text.strip()==key,revisions=revisions)
  rows.append(row);Path('/results/transport-probe.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
