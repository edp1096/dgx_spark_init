import json,pathlib,urllib.request,copy,sys,time
r=pathlib.Path(__file__).parent;kind=sys.argv[1];results=[]
for _ in range(600):
 try:
  with urllib.request.urlopen('http://127.0.0.1:18018/health',timeout=1) as f:
   if f.status==200:break
 except Exception:pass
 time.sleep(1)
else:raise RuntimeError('Trial not ready')
for thinking in [False,True]:
 for label,prompt in [('arithmetic','17 곱하기 23은? 정답 숫자만 답하세요.'),('prose','벡터 검색이 무엇인지 한국어로 세 문장으로 설명하세요.')]:
  pair=[]
  for draft in [False,True,False]:
   body={'model':'edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local','messages':[{'role':'user','content':prompt}],'max_tokens':128,'temperature':0,'stream':False,'draft':draft,'chat_template_kwargs':{'enable_thinking':thinking}}
   if thinking:body['chat_template_kwargs']['reasoning_effort']='medium'
   req=urllib.request.Request('http://127.0.0.1:18018/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=600) as f:pair.append(json.load(f))
  shas=[x.get('tensorfold',{}).get('token_sha') for x in pair]
  row={'mode':'medium' if thinking else 'off','label':label,'drafted_equal_plain':bool(all(shas)) and len(set(shas))==1,'plain_repeat_equal':bool(shas[0]) and shas[0]==shas[2],'responses':pair}
  results.append(row);(r/(kind+'-exactness.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2))
  print(json.dumps({k:v for k,v in row.items() if k!='responses'}),flush=True)
