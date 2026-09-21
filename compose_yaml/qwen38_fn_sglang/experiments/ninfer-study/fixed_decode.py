"""Fixed 256-token serving workload; ignore EOS is for timing, not chat behavior."""
import argparse,json,pathlib,time,urllib.request
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18000');p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
def post(path,body):
 r=urllib.request.Request(a.url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=300) as f:
  data=f.read()
  return data.decode() if path=="/flush_cache" else json.loads(data)
cases={'ko':'한국어로 컴퓨터 캐시가 무엇인지 예시를 들어 자세히 설명하세요.','code':'Write a Python implementation of a bounded LRU cache with get and put methods and explain it.'}
rows=[]
for name,prompt in cases.items():
 post('/flush_cache',{})
 text='<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
 for i in range(4):
  t=time.monotonic();r=post('/generate',{'text':text,'sampling_params':{'temperature':0,'top_k':-1,'top_p':1,'max_new_tokens':256,'ignore_eos':True}})
  row={'case':name,'round':i,'seconds':time.monotonic()-t,'result':r};rows.append(row)
  assert r['meta_info']['completion_tokens']==256
  print(json.dumps({'case':name,'round':i,'seconds':row['seconds']}),flush=True)
a.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
