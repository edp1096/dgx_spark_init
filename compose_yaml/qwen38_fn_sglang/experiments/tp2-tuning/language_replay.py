"""Replay an affected conversation locally; persist only aggregate diagnostics."""
import json,sqlite3,time,hashlib,urllib.request
from pathlib import Path
from measure import chat
root=Path(__file__).resolve().parent
while not (root/'results/quality-controls.json').exists():time.sleep(2)
project=next(p for p in root.parents if (p/'util/talk/dist/sparktalk.db').exists())
c=sqlite3.connect('file:'+str(project/'util/talk/dist/sparktalk.db')+'?mode=ro',uri=True)
session=c.execute('select session_id from messages where id=1740').fetchone()[0]
rows=c.execute("select id,role,content from messages where session_id=? and id<1740 and role in ('user','assistant') order by id desc limit 10",(session,)).fetchall()[::-1]
assert rows[-1][1]=='user'
history=[{'role':role,'content':text} for _,role,text in rows[:-1]];prompt=rows[-1][2]
cfg=json.load(urllib.request.urlopen('http://127.0.0.1:8585/api/config'))
base=cfg['model']['system_prompt'];rule='한국어 답변에서는 한글과 필요한 영문·숫자로 작성하고 한자 및 중국어 글자는 쓰지 마세요. 용어 뒤 괄호에 한자를 병기하지 마세요. 단, 사용자가 한자·중국어의 인용·번역·표기를 명시적으로 요청한 부분에는 해당 원문을 보존하세요.'
d={'tools_included':False,'recall_included':False,'case_user_message_id':rows[-1][0],'history_messages':len(history),'effort':'xhigh','rows':[]}
p=root/'results/language-replay-summary.json'
for repeat in range(2):
 for variant,system in [('current',base),('guide',base+'\n\n'+rule)]:
  r=chat(prompt,history=history,system=system,temperature=.7,top_p=.95,effort='xhigh',max_tokens=8192,seed=42+repeat)
  row={k:r[k] for k in ['started','finished','finish','usage','ttft','elapsed']};row.update(variant=variant,repeat=repeat,hanzi_count=len(r['hanzi']),answer_chars=len(r['text']),answer_sha256=hashlib.sha256(r['text'].encode()).hexdigest())
  d['rows'].append(row);p.write_text(json.dumps(d,ensure_ascii=False,indent=2));print(variant,repeat,'hanzi',row['hanzi_count'],'finish',row['finish'],'chars',row['answer_chars'],flush=True)
d['completed']=True;p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
