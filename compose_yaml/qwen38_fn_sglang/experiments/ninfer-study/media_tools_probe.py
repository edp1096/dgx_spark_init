"""Functional QAD regression requests; does not execute any returned tool call."""
import argparse,base64,json,pathlib,time,urllib.request
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18000');p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--image',type=pathlib.Path,required=True);p.add_argument('--video',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
def req(path,body=None):
 r=urllib.request.Request(a.url+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=180) as f:return json.load(f)
model=req('/v1/models')['data'][0]['id'];out=[]
def run(name,messages,**extra):
 start=time.monotonic();r=req('/v1/chat/completions',dict(model=model,messages=messages,temperature=0,max_tokens=128,chat_template_kwargs={'enable_thinking':False},**extra))
 out.append(dict(name=name,seconds=time.monotonic()-start,response=r));print(json.dumps(out[-1]),flush=True)
 return r['choices'][0]['message']
tools=[{'type':'function','function':{'name':'get_weather','description':'Look up weather in a city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
r=run('tool',[{'role':'user','content':'Call get_weather with city Seoul.'}],tools=tools,tool_choice={'type':'function','function':{'name':'get_weather'}})
assert r['tool_calls'][0]['function']['name']=='get_weather'
assert json.loads(r['tool_calls'][0]['function']['arguments'])['city']=='Seoul'
for name,path,mime,kind in [('image',a.image,'image/png','image_url'),('video',a.video,'video/webm','video_url')]:
 data='data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()
 prompt='Name the dominant color. Answer in English.' if name=='image' else 'Name the colors in order. Answer in English.'
 run(name,[{'role':'user','content':[{'type':'text','text':prompt},{'type':kind,kind:{'url':data}}]}])
a.output.write_text(json.dumps(out,indent=2))
