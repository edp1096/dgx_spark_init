import requests,time,json,threading,subprocess
from pathlib import Path
from PIL import Image
import measure as m
ROOT=m.ROOT
samples=[];stop=threading.Event()
def sample():
 while not stop.is_set():
  samples.append(dict(time=time.time(),**m.mem()));stop.wait(.2)
t=threading.Thread(target=sample);t.start()
try:
 for _ in range(300):
  d=requests.get('http://127.0.0.1:8585/api/runtime',timeout=20).json();op=d.get('operation',{})
  print(json.dumps({'time':time.time(),'state':op.get('state'),'phase':op.get('phase'),'progress':op.get('progress'),'memory':m.mem()}),flush=True)
  if op.get('state')=='failed':raise RuntimeError(op.get('error'))
  if op.get('state')=='complete':break
  time.sleep(5)
 else:raise RuntimeError('startup timeout')
 c=requests.get('http://127.0.0.1:8585/api/config').json()
 endpoint=c['model']['endpoint'].rstrip('/')
 payload={'model':c['model']['model'] if 'model' in c['model'] else c['model'].get('name','qwen3.8-flash-next'),'messages':[{'role':'user','content':'Reply with only the result of 7 + 8.'}],'max_tokens':32,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
 payload['model']='qwen3.8-flash-next'
 r=requests.post(endpoint+'/v1/chat/completions',json=payload,timeout=180);r.raise_for_status()
 (ROOT/'integration-chat.json').write_text(json.dumps(r.json(),indent=2))
 print('CHAT',r.json()['choices'][0]['message'],flush=True)
 src=m.dataurl(Image.open(ROOT/'baseline-generate.png').convert('RGB'));small=m.dataurl(Image.open(ROOT/'baseline-generate.png').convert('RGB').resize((768,768)))
 m.run('integration-outpaint','outpaint',anypaint_image=small,outpaint_left=128,outpaint_right=128,outpaint_top=128,outpaint_bottom=128)
 m.run('integration-object','object_remove',anypaint_image=src,mask_box=[200,280,530,800])
 m.run('integration-background','background_cleanup',source_image=src)
 m.run('integration-inpaint','inpaint',anypaint_image=src,mask_box=[200,280,530,800])
 m.run('integration-rembg','background_remove',source_image=src,background_method='lora_rembg')
 print('COMPLETE',flush=True)
finally:
 stop.set();t.join()
 (ROOT/'integration-samples.json').write_text(json.dumps(samples))
 summary={'minimum_available_gib':min(s['available_gib'] for s in samples),'maximum_swap_gib':max(s['swap_gib'] for s in samples),'initial_swap_gib':samples[0]['swap_gib'],'final':m.mem()}
 (ROOT/'integration-summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary),flush=True)
