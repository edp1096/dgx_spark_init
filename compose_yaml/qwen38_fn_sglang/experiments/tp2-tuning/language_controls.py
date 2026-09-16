import json
from pathlib import Path
from measure import chat
rule='한국어 답변에서는 한글과 필요한 영문·숫자로 작성하고 한자 및 중국어 글자는 쓰지 마세요. 용어 뒤 괄호에 한자를 병기하지 마세요. 단, 사용자가 한자·중국어의 인용·번역·표기를 명시적으로 요청한 부분에는 해당 원문을 보존하세요.'
rows=[]
for name,prompt,check in [
 ('quote','다음 중국어 원문을 그대로 인용한 뒤 한국어로 번역해 주세요: 你好，世界。',lambda s:'你好' in s and '世界' in s),
 ('hanja_requested','경복궁의 이름을 한자로 정확히 적어 주세요.',lambda s:'景福宮' in s),
 ('english','Answer only in English: explain what a cache does in two sentences.',lambda s:not any('\uac00'<=c<='\ud7a3' for c in s)),
 ('json','JSON만 출력하세요. name은 Alice, count는 숫자 3인 객체를 반환하세요.',lambda s:json.loads(s)=={'name':'Alice','count':3}),
]:
 r=chat(prompt,system=rule,temperature=.7,top_p=.95,max_tokens=256)
 try:r['passed']=check(r['text'])
 except Exception:r['passed']=False
 r['name']=name;rows.append(r);print(name,r['passed'],flush=True)
(Path(__file__).parent/'results/language-controls.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
