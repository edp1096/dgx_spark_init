import requests,subprocess,time,json,threading,base64,io,sys,os
from pathlib import Path
from PIL import Image
ROOT=Path(os.environ.get('FLUX_BENCH_OUTPUT', '/tmp/flux-lora-memory-20260913'))
ROOT.mkdir(parents=True, exist_ok=True)
API='http://127.0.0.1:8691'
G=1024**3

def mem():
 d={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())}
 return {'used_gib':(d['MemTotal']-d['MemAvailable'])/G,'available_gib':d['MemAvailable']/G,'free_gib':d['MemFree']/G,'swap_gib':(d['SwapTotal']-d['SwapFree'])/G}
def wait():
 for _ in range(240):
  try:
   if requests.get(API+'/health',timeout=2).ok:return
  except requests.RequestException:pass
  time.sleep(.5)
 raise RuntimeError('health timeout')
def restart():
 subprocess.run(['docker','restart','flux2-klein-nvfp4-api'],check=True,stdout=subprocess.DEVNULL);wait();time.sleep(1)
def gpu():
 return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
def dataurl(img):
 b=io.BytesIO();img.save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
def run(name,op,**extra):
 payload=dict(model='flux2-klein-4b-nvfp4',operation=op,prompt='Two ceramic cups on a wooden table, soft daylight, realistic photograph.',size='1024x1024',n=1,response_format='b64_json',seed=42,**extra)
 samples=[];stop=threading.Event()
 def sampler():
  while not stop.is_set():
   samples.append(dict(t=time.monotonic(),**mem()));stop.wait(.05)
 before=mem();thread=threading.Thread(target=sampler);thread.start();start=time.monotonic()
 try:
  r=requests.post(API+'/v1/images/generations',json=payload,timeout=600)
  if not r.ok:raise RuntimeError(str(r.status_code)+' '+r.text[:1200])
  raw=base64.b64decode(r.json()['data'][0]['b64_json']);out=ROOT/(name+'.png');out.write_bytes(raw)
  im=Image.open(io.BytesIO(raw));im.load()
  elapsed=time.monotonic()-start;time.sleep(2)
 finally:stop.set();thread.join()
 after=mem();peak=max(s['used_gib'] for s in samples)
 result=dict(name=name,operation=op,seconds=elapsed,before=before,after=after,peak_used_gib=peak,increment_gib=peak-before['used_gib'],gpu_after=gpu(),size=im.size,mode=im.mode)
 if im.mode=='RGBA':result['alpha_range']=im.getchannel('A').getextrema()
 (ROOT/(name+'.samples.json')).write_text(json.dumps(samples))
 (ROOT/(name+'.json')).write_text(json.dumps(result,indent=2))
 print(json.dumps(result),flush=True)
 return im

if __name__=='__main__':
 wait()
 run('baseline-generate','generate')
 source=Image.open(ROOT/'baseline-generate.png').convert('RGB');src=dataurl(source);small=dataurl(source.resize((768,768)))
 jobs=[('reference','identity_edit',{'source_image':src}),('outpaint','outpaint',{'anypaint_image':small,'outpaint_left':128,'outpaint_right':128,'outpaint_top':128,'outpaint_bottom':128}),('object','object_remove',{'anypaint_image':src,'mask_box':[200,280,530,800]}),('background','background_cleanup',{'source_image':src}),('inpaint','inpaint',{'anypaint_image':src,'mask_box':[200,280,530,800]}),('rembg','background_remove',{'source_image':src})]
 for name,op,kw in jobs:
  restart();run('isolated-'+name,op,**kw)
 restart();run('sequence-base','identity_edit',source_image=src)
 for cycle in range(3):
  for name,op,kw in jobs[1:4]:run(f'sequence-{cycle}-{name}',op,**kw)
 run('sequence-return-base','identity_edit',source_image=src)
 print('COMPLETE',flush=True)
