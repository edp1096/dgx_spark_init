"""Local serving checks and measured streaming latency; no external API calls."""
import argparse,base64,json,struct,time,urllib.request,zlib
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--long-tokens',type=int,default=0);a=p.parse_args()
a.out.parent.mkdir(parents=True,exist_ok=True)
def post(path,data,timeout=120):
 req=urllib.request.Request(a.url+path,data=json.dumps(data,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 return urllib.request.urlopen(req,timeout=timeout)
def chat(messages,limit=512,tools=None,timeout=300):
 body={'model':a.model,'messages':messages,'temperature':0,'max_tokens':limit,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
 if tools:body.update(tools=tools,tool_choice='auto')
 start=time.monotonic();first=None;text='';reasoning='';calls={};usage={}
 with post('/v1/chat/completions',body,timeout) as response:
  for raw in response:
   if not raw.startswith(b'data:'):continue
   raw=raw[5:].strip()
   if raw==b'[DONE]':break
   event=json.loads(raw)
   if event.get('usage'):usage=event['usage']
   for choice in event.get('choices',[]):
    delta=choice.get('delta',{})
    if first is None and (delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning') or delta.get('tool_calls')):first=time.monotonic()-start
    text+=delta.get('content') or '';reasoning+=delta.get('reasoning_content') or delta.get('reasoning') or ''
    for call in delta.get('tool_calls',[]):
     target=calls.setdefault(call['index'],{'name':'','arguments':''});f=call.get('function',{})
     target['name']+=f.get('name') or '';target['arguments']+=f.get('arguments') or ''
 elapsed=time.monotonic()-start;tokens=usage.get('completion_tokens',0)
 return {'content':text,'reasoning':reasoning,'tool_calls':list(calls.values()),'usage':usage,'seconds':elapsed,'ttft_seconds':first,'tg_tokens_per_second':(tokens-1)/(elapsed-first) if first is not None and elapsed>first and tokens>1 else None}
def user(text):return [{'role':'user','content':text}]
results={'model':a.model,'endpoint':a.url,'started_at':time.time(),'checks':[]}
def save():a.out.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
def check(name,messages,predicate,**kwargs):
 result=chat(messages,**kwargs);result.update(name=name,passed=bool(predicate(result)));results['checks'].append(result);save();print(name,result['passed'],'ttft',result['ttft_seconds'],'tg',result['tg_tokens_per_second'],flush=True)
 return result
check('warmup',user('안녕이라고만 답해.'),lambda r:bool(r['content']),limit=64)
check('arithmetic',user('17 곱하기 23의 답을 숫자로만 써라.'),lambda r:'391' in r['content'],limit=128)
check('korean',user('비가 오는 날 산책할 때 준비할 것을 한국어로 세 문장 작성해라.'),lambda r:sum('가'<=c<='힣' for c in r['content'])>=15,limit=384)
check('code',user('Python으로 두 정수 a,b의 합을 반환하는 add 함수를 작성해라.'),lambda r:'def add(' in r['content'] and '+' in r['content'],limit=256)
tools=[{'type':'function','function':{'name':'lookup_palace','description':'Look up a Korean palace.','parameters':{'type':'object','properties':{'palace':{'type':'string'}},'required':['palace']}}}]
def tool_ok(r):
 for call in r['tool_calls']:
  try:
   if call['name']=='lookup_palace' and json.loads(call['arguments']).get('palace')=='경복궁':return True
  except ValueError:pass
 return False
check('tool_call',user('lookup_palace 도구를 palace="경복궁"으로 호출해라.'),tool_ok,tools=tools,limit=256)
# A tiny generated red PNG, avoiding external media or image downloads.
def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',32,32,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+b'\xff\0\0'*32)*32))+chunk(b'IEND',b'')
check('vision',[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}},{'type':'text','text':'이 이미지의 주된 색을 한국어로 답해라.'}]}],lambda r:any(word in r['content'] for word in ['빨','붉','적색']),limit=128)
if a.long_tokens:
 target=a.long_tokens
 def messages(n):
  filler='이 문장은 문맥 길이를 확인하기 위한 일반적인 기록이다. '
  return user('세 암호를 기억하고 마지막 질문에 답해라. 암호 A는 482913이다.\n'+filler*(n//2)+'\n암호 B는 571026이다.\n'+filler*(n-n//2)+'\n암호 C는 839405이다.\n암호 A, B, C를 순서대로 JSON 배열에 써라. 설명은 하지 마라.')
 count=1000
 for _ in range(8):
  with post('/tokenize',{'model':a.model,'messages':messages(count),'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}},300) as response:tokenized=json.load(response)
  actual=tokenized.get('count',len(tokenized.get('tokens',[])))
  if target-128<=actual<=target:break
  count=max(1,count+int((target-actual)/(actual/count))-1)
 else:raise RuntimeError('Could not size the long input safely')
 results['long_input_tokens']=actual;save()
 check('long_retrieval',messages(count),lambda r:all(value in r['content'] for value in ['482913','571026','839405']),limit=512,timeout=7200)
results['finished_at']=time.time();results['passed']=all(c['passed'] for c in results['checks']);save()

if not results['passed']:raise SystemExit(1)
