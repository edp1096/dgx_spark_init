from measure import chat,flush_cache
from pathlib import Path
import json
rows=[];p=Path(__file__).parent/'results/chunk2048-failure-recheck.json'
for repeat in range(4):
 count=1800;key='KEY_1800_0';lines=['Record %06d: ordinary filler text.'%i for i in range(count)];lines.insert(count//2,'The secret retrieval key is '+key+'.');prompt='Read the records and return only the secret retrieval key.\n'+'\n'.join(lines)+'\nReturn the secret retrieval key only.'
 flush_cache();r=chat(prompt,max_tokens=32);r['passed']=r['text'].strip()==key;rows.append(r);p.write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(repeat,r['passed'],repr(r['text']),flush=True)
