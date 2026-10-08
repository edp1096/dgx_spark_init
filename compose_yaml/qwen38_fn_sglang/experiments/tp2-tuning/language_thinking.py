from pathlib import Path
import json,time
from measure import chat
rule='한국어 답변에서는 한글과 필요한 영문·숫자로 작성하고 한자 및 중국어 글자는 쓰지 마세요. 용어 뒤 괄호에 한자를 병기하지 마세요. 단, 사용자가 한자·중국어의 인용·번역·표기를 명시적으로 요청한 부분에는 해당 원문을 보존하세요.'
base='친근한 한국어 반말로 답하세요.'
prompts=[('palaces','조선 시대 서울의 다섯 궁궐을 창건 시기, 용도, 대표 건물 기준으로 한국어 표로 비교하고 배치 원리를 간결하게 설명해 줘.'),('architecture','한국 궁궐의 정궁과 이궁의 차이, 문과 조정의 배치 원리를 간결하게 설명해 줘.')]
p=Path(__file__).parent/'results/language-thinking.json';d={'effort':'xhigh','rows':[]}
for repeat in range(2):
 for name,prompt in prompts[:1]:
  for variant,system in [('current',base),('guide',base+'\n\n'+rule)]:
   r=chat(prompt,system=system,temperature=.7,top_p=.95,effort='xhigh',max_tokens=8192,seed=42+repeat)
   r.update(name=name,variant=variant,repeat=repeat);d['rows'].append(r);p.write_text(json.dumps(d,ensure_ascii=False,indent=2));print(name,variant,repeat,'hanzi',len(r['hanzi']),'finish',r['finish'],'tokens',r['usage'].get('completion_tokens'),flush=True)
d['completed']=True;p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
