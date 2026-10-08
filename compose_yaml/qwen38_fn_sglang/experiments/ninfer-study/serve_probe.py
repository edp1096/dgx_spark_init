import argparse,json,time,urllib.request,pathlib
parser=argparse.ArgumentParser()
parser.add_argument('--label',required=True)
parser.add_argument('--output-dir',type=pathlib.Path,required=True)
parser.add_argument('--url',default='http://127.0.0.1:18000')
args=parser.parse_args();mode=args.label;base=args.url;folder=args.output_dir
folder.mkdir(parents=True,exist_ok=True)
if (folder/(mode+'-responses.json')).exists():raise FileExistsError('Use a fresh label/output directory')
def req(path,body=None):
 r=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=900) as f:
  data=f.read()
  return json.loads(data) if data else {}
for _ in range(120):
 try:req('/health');break
 except Exception:time.sleep(5)
else:raise RuntimeError('model did not become healthy')
model=req('/v1/models')['data'][0]['id']; results=[]
cases=[('ko','한국어로 답하세요. 17 곱하기 19의 결과를 말하고, 곱셈 계산 과정을 짧게 설명하세요.'),('code','Write only a Python function implementing binary search on a sorted list, returning -1 if absent.'),('json','Return only JSON with keys name, count, enabled. Values are sample, 7, true.'),('prose','Explain why leaves change color in autumn in a short paragraph.')]
for name,text in [('warmup','Reply OK.')]+cases+[(n+'_repeat',t) for n,t in cases]:
 t=time.monotonic();r=req('/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':text}],'temperature':0,'max_tokens':256,'chat_template_kwargs':{'enable_thinking':False}})
 row={'name':name,'seconds':time.monotonic()-t,'response':r};results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
# Repeated prefix, then branch: different suffixes must not pick up stale GDN state.
prefix='Secret marker: BLUE-731.\n'+('The notebook lists ordinary household objects.\n'*4000)
for name,tail in [('long','What is the secret marker? Answer only the marker.'),('long_reuse','What is the secret marker? Answer only the marker.'),('branch','Ignore the marker. What is 23 plus 19? Answer only the number.')]:
 t=time.monotonic();r=req('/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':prefix+tail}],'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}})
 row={'name':name,'seconds':time.monotonic()-t,'response':r};results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
(folder/(mode+'-responses.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2))
(folder/(mode+'-server-info.json')).write_text(json.dumps(req('/get_server_info'),indent=2))
