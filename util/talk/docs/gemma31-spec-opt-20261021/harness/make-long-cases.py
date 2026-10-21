from pathlib import Path
import json,hashlib
from transformers import AutoTokenizer
r=Path('/job')
t=AutoTokenizer.from_pretrained(r/'models/target',local_files_only=True)
cases=json.loads((r/'cases.json').read_text())
filler='기록에는 일반적인 운영 상태와 오류 없는 처리 결과가 반복됩니다. '
prefix='문서의 비밀 암호는 청록펭귄이다.\n'
suffix='\n답변 첫 줄에 문서 앞의 비밀 암호를 적으세요. 이어서 운영 기록을 검토할 때 확인해야 할 사항을 한국어로 구체적으로 설명하세요.'
def build(n):
    prompt=prefix+filler*n+suffix
    ids=t.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    assert isinstance(ids,list) and all(isinstance(x,int) for x in ids)
    return prompt,ids
for target in [32768,65536]:
    lo,hi=1,target//max(1,len(t.encode(filler,add_special_tokens=False)))+128
    while len(build(hi)[1])<=target:hi*=2
    while lo<hi:
        mid=(lo+hi+1)//2
        if len(build(mid)[1])<=target:lo=mid
        else:hi=mid-1
    prompt,ids=build(lo)
    case=dict(name=f'long_ko_{target//1024}k',kind='long_speed_ko',input_ids=ids,
              max_new_tokens=512,ignore_eos=True,prompt=prompt,expected='청록펭귄')
    cases.append(case);print(case['name'],len(ids),hashlib.sha256(json.dumps(ids).encode()).hexdigest(),flush=True)
(r/'cases-with-long.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
