import json,urllib.request,time,pathlib,hashlib,re,sys,uuid,statistics,os
ROOT=pathlib.Path(__file__).parent;MODEL='edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local'
ENGINE=sys.argv[1];URL='http://127.0.0.1:'+('18016' if ENGINE=='prepare' else '18014' if ENGINE=='vllm' else '18018');ROWS=[]
def save(name,data):
 p=ROOT/(name+'.next');p.write_text(json.dumps(data,ensure_ascii=False,indent=2));p.replace(ROOT/name)
def phase(stage,**kw):
 x={'engine':ENGINE,'stage':stage,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw};save('bench-state.json',x);print(json.dumps(x,ensure_ascii=False),flush=True)
def post(path,body):
 req=urllib.request.Request(URL+path,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as r:return json.load(r)
def tokenize(body):
 b={k:body[k] for k in ['model','messages','chat_template_kwargs']};b['add_generation_prompt']=True
 x=post('/tokenize',b);ids=x['tokens'];return len(ids),hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def make_body(text,mode,cap):
 kw={'enable_thinking':mode=='medium'}
 if mode=='medium':kw['reasoning_effort']='medium'
 return {'model':MODEL,'messages':[{'role':'user','content':text}],'chat_template_kwargs':kw,'temperature':0,'max_tokens':cap,'stream':True,'stream_options':{'include_usage':True}}
def metrics():
 data=urllib.request.urlopen(URL+'/metrics',timeout=10).read().decode();out={}
 for line in data.splitlines():
  m=re.fullmatch(r'([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)',line)
  if m:
   try:out[m[1]]=out.get(m[1],0)+float(m[2])
   except ValueError:pass
 return out
def run(case):
 body=case['body'];before=metrics() if ENGINE=='vllm' else {};start=time.monotonic();first=None;answer_first=None;last=None;content=[];reasoning=[];native={};usage={};finish=None
 req=urllib.request.Request(URL+'/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=1800) as r:
  for line in r:
   line=line.decode().strip()
   if not line.startswith('data: {'):continue
   x=json.loads(line[6:]);native=x.get('tensorfold') or native;usage=x.get('usage') or usage
   if x.get('error'):raise RuntimeError(str(x['error']))
   for choice in x.get('choices',[]):
    finish=choice.get('finish_reason') or finish;d=choice.get('delta') or {};a=d.get('content') or '';thought=d.get('reasoning_content') or d.get('reasoning') or ''
    if a or thought:
     now=time.monotonic();first=first or now;last=now
     if a:answer_first=answer_first or now;content.append(a)
     if thought:reasoning.append(thought)
 elapsed=time.monotonic()-start;n=usage.get('completion_tokens',native.get('completion_tokens',native.get('completion',0)));p=usage.get('prompt_tokens',native.get('prompt_tokens',native.get('prompt',0)));cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',native.get('cached_tokens',native.get('cached')))
 native_deltas={}
 if ENGINE=='vllm':
  for i in range(200):
   after=metrics();native_deltas={k:after.get(k,0)-before.get(k,0) for k in set(after)|set(before)}
   if native_deltas.get('vllm:request_decode_time_seconds_count',0)>=1:break
   time.sleep(.05)
  assert native_deltas.get('vllm:request_decode_time_seconds_count',0)==1,native_deltas.get('vllm:request_decode_time_seconds_count')
  pre=native_deltas['vllm:request_prefill_time_seconds_sum'];decode=native_deltas['vllm:request_decode_time_seconds_sum']
  if cached is None:cached=int(native_deltas.get('vllm:prefix_cache_hits_total',native_deltas.get('vllm:prefix_cache_hits',0)))
 else:
  pre=native.get('prefill_seconds',native.get('prefill_s'));decode=native.get('decode_seconds',native.get('decode_s'))
  if decode is None and native.get('tokens_per_second',0)>0:decode=(n-1)/native['tokens_per_second']
 save(ENGINE+'-last-response.json',{'case':case['id'],'usage':usage,'native':native,'answer':''.join(content),'reasoning_chars':sum(len(x) for x in reasoning)})
 assert n>0 and p>0 and pre is not None and decode is not None,(usage,native)
 answer=''.join(content);thought=''.join(reasoning)
 row={'engine':ENGINE,'case':case['id'],'kind':case['kind'],'mode':case['mode'],'prompt_tokens':p,'completion_tokens':n,'cached_tokens':cached,'prefill_s':pre,'decode_s':decode,'pp':(p-(cached or 0))/pre if pre>0 else None,'tg':(n-1)/decode if n>1 and decode>0 else None,'client_tg':(n-1)/(last-first) if first and last and last>first else None,'ttft_s':first-start if first else None,'ttfa_s':answer_first-start if answer_first else None,'elapsed_s':elapsed,'finish_reason':finish,'has_final_answer':bool(answer.strip()),'answer':answer,'reasoning_chars':len(thought),'reasoning_sha256':hashlib.sha256(thought.encode()).hexdigest(),'answer_sha256':hashlib.sha256(answer.encode()).hexdigest(),'native':native,'usage':usage,'reasoning_tokens':(usage.get('completion_tokens_details') or {}).get('reasoning_tokens',native.get('reasoning_tokens')),'hangul_answer_chars':sum('가'<=c<='힣' for c in answer),'metrics_deltas':{k:v for k,v in native_deltas.items() if v and ('time_seconds_' in k or 'token' in k or 'prefix' in k)},'max_tokens':body['max_tokens']}
 if case['kind']=='warmup':row['arithmetic_correct']='391' in answer;row['integer_only']=answer.strip()=='391'
 ROWS.append(row);save(ENGINE+'-rows.json',ROWS);print(json.dumps({k:row[k] for k in ['case','mode','prompt_tokens','completion_tokens','pp','tg','ttft_s','ttfa_s','reasoning_chars','has_final_answer']},ensure_ascii=False),flush=True)
 return row

def build_cases():
 cases=[]
 for mode in ['off','medium']:
  cases.append({'id':'warmup-'+mode,'kind':'warmup','mode':mode,'body':make_body('17 곱하기 23은? 정답 숫자만 답하세요.',mode,512)})
 prompts=[('explain','벡터 검색과 키워드 검색의 차이를 처음 배우는 사람에게 한국어로 설명하세요. 각각의 장단점과 함께 사용하는 이유를 포함해 여섯 문장 이상 작성하세요.',256),('prose','비가 그친 저녁 동네 도서관을 나서는 사람의 하루를 한국어 소설 문체로 여덟 문장 이상 묘사하세요. 목록이나 제목은 쓰지 마세요.',256),('practical','두 대의 컴퓨터가 하나의 작업을 나누어 처리할 때 통신 속도가 왜 중요한지 한국어로 설명하세요. 쉬운 비유와 실제 상황을 포함해 여섯 문장 이상 작성하세요.',256),('long','한국어로 긴 설명문을 작성하세요. 주제는 두 대의 소형 AI 컴퓨터를 연결해 큰 언어 모델을 실행하는 과정입니다. 모델 가중치와 KV 캐시의 차이, 토큰을 입력하고 생성하는 단계, 메모리와 네트워크의 병목, 생성 속도를 공정하게 비교하는 방법을 차례로 설명하세요. 초보자가 이해할 수 있는 비유를 사용하고 각 주제를 충분한 분량의 문단으로 자세히 설명하세요. 최소 열두 문단을 작성하고 제목과 목록은 사용하지 마세요.',768)]
 for rep in range(2):
  for mode in ['off','medium']:
   for label,text,cap in prompts:
    nonce=hashlib.sha256(f'compare:{rep}:{mode}:{label}'.encode()).hexdigest()[:24]
    cases.append({'id':f'tg-{label}-{mode}-{rep}','kind':'tg','mode':mode,'body':make_body(f'작업 ID {nonce}. ID를 답변에 쓰지 마세요.\n'+text,mode,2048 if mode=='medium' else cap)})
 for mode in ['off','medium']:
  for target in [4096,16384,131072]:
   nonce=hashlib.sha256(f'prefill:{mode}:{target}'.encode()).hexdigest()[:24]
   records=[f'기록 {i:05d}: 연결={hashlib.sha256(f"{nonce}:{i}".encode()).hexdigest()[:12]}, 지연={i%83+5}ms. 서버는 요청을 처리했고 저장 결과는 정상입니다. 다음 점검에는 연결 상태와 메모리 사용량을 확인합니다.\n' for i in range(4000)]
   def text(n,pad=0):return f'작업 ID {nonce}. 다음은 합성 운영 기록입니다.\n'+''.join(records[:n])+(' x'*pad)+'\n위 기록의 성격을 한국어 한 문장으로 설명하세요.'
   count,_=tokenize(make_body(text(100),mode,16));num=int((target-100)*100/count)
   for attempt in range(12):
    count,_=tokenize(make_body(text(num),mode,16))
    if 0<=target-count<250:break
    num+=(-max(1,int((count-target)*num/count)+1) if count>target else max(1,int((target-count-100)*num/count)))
   else:raise RuntimeError('Prompt sizing failed')
   pad=target-count
   for adjust in range(8):
    body=make_body(text(num,pad),mode,16);count,sha=tokenize(body)
    if count==target:break
    pad+=target-count
    if pad<0:
     num-=1;count,_=tokenize(make_body(text(num),mode,16));pad=target-count
   assert count==target,(target,count)
   cases.append({'id':f'pp-{target}-{mode}','kind':'pp','mode':mode,'body':body,'expected_prompt_tokens':count,'token_ids_sha256':sha})
 for case in cases:
  if 'token_ids_sha256' not in case:
   count,sha=tokenize(case['body']);case.update(expected_prompt_tokens=count,token_ids_sha256=sha)
 save('cases.json',cases);return cases

if ENGINE=='prepare':
 build_cases();phase('workload-ready');sys.exit(0)
phase('preparing-workload')
cases=json.loads((ROOT/'cases.json').read_text()) if (ROOT/'cases.json').exists() else build_cases()
if os.environ.get('BENCH_CASE_IDS'):
 selected=set(os.environ['BENCH_CASE_IDS'].split(','));cases=[x for x in cases if x['id'] in selected]
 assert len(cases)==len(selected),'Unknown benchmark case requested'
for case in cases:
 phase('checking-tokenization',case=case['id']);count,sha=tokenize(case['body']);assert count==case['expected_prompt_tokens'] and sha==case['token_ids_sha256'],('Template or tokenizer mismatch',case['id'],count,case['expected_prompt_tokens'],sha,case['token_ids_sha256'])
 phase('measuring',case=case['id'],mode=case['mode']);run(case)
phase('complete',requests=len(ROWS))
