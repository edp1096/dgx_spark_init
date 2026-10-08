import argparse,json,time,re,statistics,urllib.request,concurrent.futures,subprocess
from pathlib import Path
URL='http://127.0.0.1:8012';MODEL='edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4'
PROMPTS=[('ko_tech','리눅스의 프로세스와 스레드를 메모리 공유, 오류 격리, 웹 서버 예시로 한국어로 설명해 주세요.'),('ko_story','비 오는 날 서점에서 자기 이름이 적힌 편지를 발견한 사람의 짧은 이야기를 한국어로 써 주세요.'),('ko_plan','작은 도서관 대출 시스템의 회원, 대출, 반납, 연체 처리 설계를 한국어로 설명하세요.'),('ko_compare','가정용 유선 네트워크와 무선 네트워크를 지연과 안정성, 설치 비용 기준으로 한국어로 비교하세요.'),('code','Write a Python function unique_stable(items) that removes duplicates preserving order for hashable items. Return only executable Python code.'),('json','Return only JSON: {"answer":42,"items":[1,2,3]}')]
RULE='한국어로 답하고, 요청하지 않은 한자·중국어 표현을 섞지 않는다. 고유명사·인용·번역 대상은 원문을 유지한다.'
def get(path):return json.load(urllib.request.urlopen(URL+path,timeout=10))
def flush_cache():
 for _ in range(60):
  try:
   urllib.request.urlopen(URL+'/flush_cache',timeout=10).read();return
  except urllib.error.HTTPError as e:
   if e.code!=400:raise
   time.sleep(.5)
 raise RuntimeError('Requests remained active; refusing a contaminated measurement')
def verify_count():
 text=urllib.request.urlopen(URL+'/metrics',timeout=10).read().decode()
 return sum(float(line.rsplit(' ',1)[1]) for line in text.splitlines() if line.startswith('sglang:spec_verify_calls_total{'))
def chat(prompt,temperature=0,top_p=.95,system=None,max_tokens=256,seed=42,tools=None,effort=None,history=None):
 messages=[]
 if system:messages.append({'role':'system','content':system})
 if history:messages.extend(history)
 messages.append({'role':'user','content':prompt})
 body=dict(model=MODEL,messages=messages,temperature=temperature,top_p=top_p,top_k=20,presence_penalty=0,seed=seed,max_tokens=max_tokens,stream=True,stream_options={'include_usage':True},chat_template_kwargs={'enable_thinking':False})
 if effort:
  body['chat_template_kwargs']['enable_thinking']=True;body['reasoning_effort']=effort
 if tools:body.update(tools=tools,tool_choice='auto')
 req=urllib.request.Request(URL+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 started=time.time();before_verify=verify_count()
 t=time.monotonic();first=None;content='';reason='';calls=[];usage={};finish=None
 with urllib.request.urlopen(req,timeout=600) as r:
  for line in r:
   if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
   d=json.loads(line[6:]);usage=d.get('usage') or usage
   if d.get('error'):raise RuntimeError(d['error'])
   for c in d.get('choices',[]):
    v=c.get('delta',{});text=v.get('content') or '';thought=v.get('reasoning_content') or '';tc=v.get('tool_calls') or []
    if (text or thought or tc) and first is None:first=time.monotonic()
    content+=text;reason+=thought;calls+=tc;finish=c.get('finish_reason') or finish
 end=time.monotonic();n=usage.get('completion_tokens',0)
 verifies=verify_count()-before_verify
 return dict(started=started,finished=time.time(),verify_calls=verifies,accept_length=max(n-1,0)/verifies if verifies else None,text=content,reasoning=reason,calls=calls,usage=usage,finish=finish,ttft=first-t if first else None,tg=(n-1)/(end-first) if first and n>1 else None,elapsed=end-t,hanzi=re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]',content))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['speed','language','language-strict','prefill'],required=True);a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
 d={'mode':a.mode,'server':get('/get_server_info'),'rows':[]}
 def save():a.out.write_text(json.dumps(d,ensure_ascii=False,indent=2))
 save()
 cases=[]
 if a.mode=='speed':
  for repeat in range(2):
   for name,prompt in PROMPTS:cases.append((name,repeat,prompt,dict()))
 elif a.mode in ('language','language-strict'):
  language_prompts=[('ko_palaces','조선 시대 서울의 다섯 궁궐을 창건 시기, 용도, 대표 건물 기준으로 한국어 표로 비교하고 배치 원리를 설명해 줘.'),('ko_architecture','한국 궁궐의 정궁과 이궁의 차이, 문과 조정의 배치 원리를 친근한 한국어로 설명해 줘.'),('ko_edit','찻잔 사진에서 컵 하나만 없애고 창가 조명과 그림자는 유지하는 편집을 마쳤다고 가정하자. 결과를 사용자에게 친근한 한국어로 설명해 줘.'),('ko_tech','작은 배치에서는 메모리 읽기량을 줄여도 속도가 개선되지 않을 수 있는 이유를 한국어로 설명해 줘.')]
  variants=[('current',.7,.95,None),('p08',.7,.8,None),('rule',.7,.95,RULE),('rule_p08',.7,.8,RULE)]
  if a.mode=='language-strict':
   strict='한국어 답변에서는 한글과 필요한 영문·숫자로 작성하고 한자 및 중국어 글자는 쓰지 마세요. 용어 뒤 괄호에 한자를 병기하지 마세요. 단, 사용자가 한자·중국어의 인용·번역·표기를 명시적으로 요청한 부분에는 해당 원문을 보존하세요.'
   variants=[('current',.7,.95,None),('strict',.7,.95,strict),('strict_p08',.7,.8,strict),('strict_t05_p08',.5,.8,strict)]
  for repeat in range(2 if a.mode=='language-strict' else 3):
   for name,prompt in language_prompts:
    for variant,temp,top,rule in variants:cases.append((name+'_'+variant,repeat,prompt,dict(temperature=temp,top_p=top,system=rule,max_tokens=384,seed=42+repeat)))
 else:
  for repeat in range(2):
   for count in [1800,7200,14400]:
    key='KEY_%d_%d'%(count,repeat);lines=['Record %06d: ordinary filler text.'%i for i in range(count)];lines.insert(count//2,'The secret retrieval key is '+key+'.');prompt='Read the records and return only the secret retrieval key.\n'+'\n'.join(lines)+'\nReturn the secret retrieval key only.';cases.append(('needle_'+str(count),repeat,prompt,dict(max_tokens=32)))
 for name,repeat,prompt,kwargs in cases:
  flush_cache()
  r=chat(prompt,**kwargs);r.update(name=name,repeat=repeat,parameters=kwargs)
  d['rows'].append(r);save();print(name,repeat,'ttft',round(r['ttft'] or 0,3),'tg',round(r['tg'] or 0,2),'hanzi',len(r['hanzi']),flush=True)
 d['completed']=True;save()
if __name__=='__main__':main()
