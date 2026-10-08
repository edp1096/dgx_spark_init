import json,random
from pathlib import Path
from measure import chat,flush_cache,get
rng=random.Random(91845);root=Path(__file__).resolve().parent
out=root/'results/selected-stress.json';d={'server':get('/get_server_info'),'rows':[]}
for repeat in range(2):
 for count in [143,290,580,1170,1800,3600]:
  key='KEY_'+''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789',k=14))
  lines=['Record %06d: ordinary filler text.'%i for i in range(count)];lines.insert(count//2,'The secret retrieval key is '+key+'.')
  prompt='Read the records and return only the secret retrieval key.\n'+'\n'.join(lines)+'\nReturn the secret retrieval key only.'
  flush_cache();r=chat(prompt,max_tokens=48);r.update(count=count,repeat=repeat,expected=key,passed=r['text'].strip()==key)
  d['rows'].append(r);out.write_text(json.dumps(d,ensure_ascii=False,indent=2));print(count,repeat,r['passed'],repr(r['text']),flush=True)
d['completed']=True;d['passed']=all(r['passed'] for r in d['rows']);out.write_text(json.dumps(d,ensure_ascii=False,indent=2))
raise SystemExit(0 if d['passed'] else 1)
