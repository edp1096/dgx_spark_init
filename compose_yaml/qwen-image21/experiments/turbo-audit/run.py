"""Isolated ComfyUI image comparison; production containers/config are preserved."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from PIL import Image,ImageDraw
H=Path(__file__).resolve().parent
WORK=H.parents[3]
ROOT=Path('/home/edp1096/.cache/model-download-jobs/qwim21-turbo-audit-20261010')
HF=Path.home()/'.cache/huggingface'
NAME='qwim21-turbo-audit'
IMAGE='sparktalk-qwim-mmh3:nvfp4-sol-i2v1'
SIGMAS=[1.0,.978453,.95418,.926626,.89508,.845148,.704534,.414568,0.0]
BASE='/hf/hub/models--abenzerps--Qwen-Image-2.1-Uncensored-GGUF/snapshots/6b34e59458d3eb7ba6a6f86a116aed5253dc02c3/qwen-image-2.1-UC-NVFP4.safetensors'
AUX='/hf/hub/models--Comfy-Org--Qwen-Image-2.1/snapshots/cb504a4090723e43f17ad01cec0359490e2de613'
CASES=[
 ('penguin','A beautiful adorable fluffy baby penguin standing on Antarctic snow, cinematic soft morning light, realistic feather texture, expressive eyes, full body, tasteful composition, no text.',20261010),
 ('hands','A candid editorial photograph of an adult East Asian woman holding a transparent glass cup of tea with both hands. Both hands and every finger are clearly visible, anatomically correct, natural warm window light, detailed skin, realistic proportions, no text.',20261010),
 ('counting','A clean product photograph on a white table. Exactly three red apples in a row on the LEFT, exactly two blue ceramic cups in a row on the RIGHT. No other objects. All five objects are separated and fully visible. Straight-on view.',20261010),
 ('korean','정사각형 한국어 카페 포스터. 크림색 배경과 짙은 갈색 타이포그래피. 위쪽에 정확히 "오늘의 커피", 가운데에 커피잔 하나, 아래쪽에 정확히 "3,500원". 그 외 글자 없이 깔끔한 편집 디자인. 한글과 숫자를 선명하고 정확하게 표시.',20261010),
 ('english','A square minimalist technology event poster with a dark navy background and crisp white and cyan typography. Exact text at the top: "SPARK LAB". Three cards in one horizontal row labelled exactly "GPU", "API", "SQL". A footer reads exactly "864 × 480 · 24 FPS". Clean professional graphic design, no other text.',20261010),
 ('alpha','A photorealistic red ceramic coffee mug with a white interior, handle on the right, isolated product cutout, centered full object, native RGBA transparent background, no shadow outside the object, no text.',20261010),
 ('counting-seed2','A clean product photograph on a white table. Exactly three red apples in a row on the LEFT, exactly two blue ceramic cups in a row on the RIGHT. No other objects. All five objects are separated and fully visible. Straight-on view.',42),
 ('korean-seed2','정사각형 한국어 카페 포스터. 크림색 배경과 짙은 갈색 타이포그래피. 위쪽에 정확히 "오늘의 커피", 가운데에 커피잔 하나, 아래쪽에 정확히 "3,500원". 그 외 글자 없이 깔끔한 편집 디자인. 한글과 숫자를 선명하고 정확하게 표시.',42),
 ('edit','Edit this illustration: change only the BLUE CIRCLE on the left to a GREEN CIRCLE. Preserve the red square on the right, its position and size, and the white background. Keep the same clean flat vector illustration style. No text.',20261010),
 ('multiref','Combine both reference images into one clean flat illustration on a white background: place the BLUE CIRCLE and RED SQUARE from the first reference on the left side, and the YELLOW TRIANGLE from the second reference on the right side. Preserve all three exact colors and shapes. No additional objects or text.',20261010),
]

def command(*cmd):return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
def save(p,d):
 tmp=p.with_name(p.name+'.hosttmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def status(profile,stage):save(ROOT/'current.json',{'profile':profile,'stage':stage,'container':NAME,'time':time.time()})
def selected_path(name):
 d=json.loads((ROOT/('download-'+name)/'metadata.json').read_text());e=d['siblings'][0]
 return '/hf/hub/models--'+d['id'].replace('/','--')+'/snapshots/'+d['sha']+'/'+e['rfilename']

def fixtures():
 p=ROOT/'inputs';p.mkdir(exist_ok=True)
 im=Image.new('RGB',(1024,1024),'white');d=ImageDraw.Draw(im);d.ellipse((120,340,400,620),fill=(20,70,235));d.rectangle((620,340,900,620),fill=(225,30,35));im.save(p/'shapes.png')
 im=Image.new('RGB',(1024,1024),'white');ImageDraw.Draw(im).polygon([(512,220),(250,780),(774,780)],fill=(245,195,20));im.save(p/'triangle.png')

def setup(profile,model,turbo,joint=False):
 out=ROOT/profile;out.mkdir(exist_ok=True)
 for n in ['requests','results','output','cancel','state']:(out/n).mkdir(exist_ok=True)
 for n in ['ready.json','worker-state.json','stop']:(out/n).unlink(missing_ok=True)
 dits={'qwim':model}
 if joint:dits['h3']='/hf/hub/models--lilcheaty--MiniMax-H3-NVFP4/snapshots/8c5abfed61e1b6a170240792b65253fba1a65b7b/minimax_h3_fl2va_pruned_nvfp4.safetensors'
 settings={'aux_loading':'dynamic','model_paths':{'text_encoders':['/opt/ComfyUI/models/text_encoders'],'vae':['/opt/ComfyUI/models/vae']},'dits':dits,'turbo':turbo,'sigmas':SIGMAS,'cache_dtype':'int8','fp8_scale_compat':model.endswith('/qwen-image-2.1-turbo-fp8.safetensors')};save(out/'settings.json',settings)
 prepare="""import os
from pathlib import Path
base=Path('"""+AUX+"""')
for kind,name in [('text_encoders','qwen3vl_8b_w4a8.safetensors'),('vae','qwen_image_2.1_vae_bf16.safetensors')]:
 p=Path('/opt/ComfyUI/models')/kind/name;p.parent.mkdir(parents=True,exist_ok=True);p.unlink(missing_ok=True);p.symlink_to(base/kind/name)
"""
 (out/'prepare.py').write_text(prepare)
 cmd=['docker','run','-d','--name',NAME,'--gpus','all','--memory','48g','--memory-swap','48g','--network','none','--ipc','host','-v',str(HF)+':/hf:ro','-v',str(out)+':/job','-v',str(ROOT/'inputs')+':/inputs:ro','-v',str(H/'worker.py')+':/audit/worker.py:ro','-e','JOB_DIR=/job','-e','IMAGE_INPUT_DIR=/inputs','-e','HF_HUB_OFFLINE=1','-e','COMFY_KITCHEN_BACKEND=cuda','-e','PYTHONUNBUFFERED=1','--entrypoint','/bin/bash',IMAGE,'-c','python /job/prepare.py && exec python /audit/worker.py']
 save(out/'docker-command.json',cmd);status(profile,'startup');command(*cmd)
 start=time.monotonic()
 while not (out/'ready.json').exists():
  state=json.loads(command('docker','inspect',NAME))[0]['State']
  if not state['Running'] or time.monotonic()-start>600:
   raise RuntimeError('Worker did not become ready: '+command('docker','logs',NAME)[-6000:])
  if (ROOT/'memory-floor.json').exists():raise RuntimeError('Memory floor reached')
  time.sleep(.5)
 save(out/'startup.json',{'seconds':time.monotonic()-start,'state':json.loads((out/'ready.json').read_text())});return out

def request(out,name,prompt,seed,cache=False,width=1024,height=1024):
 status(out.name,name)
 refs=['shapes.png'] if name=='edit' else ['shapes.png','triangle.png'] if name.startswith('multiref') else ['hands-reference.png'] if name=='photo-edit' else []
 req={'case':name,'prompt':prompt,'seed':seed,'width':width,'height':height,'reference_files':refs,'operation':'edit' if refs else 'generate','conditioning_cache':cache}
 target=out/'requests'/(name+'.json');tmp=target.with_suffix('.tmp');save(tmp,req);tmp.replace(target)
 start=time.monotonic();dest=out/'results'/target.name
 while not dest.exists():
  state=json.loads(command('docker','inspect',NAME))[0]['State']
  if not state['Running'] or time.monotonic()-start>900:raise RuntimeError('Generation worker failed: '+command('docker','logs',NAME)[-6000:])
  if (ROOT/'memory-floor.json').exists():raise RuntimeError('Memory floor reached')
  time.sleep(.5)
 result=json.loads(dest.read_text());print(out.name,name,result['status'],round(result.get('seconds',0),2),flush=True)
 if result['status']!='success':raise RuntimeError(result)
 im=Image.open(out/'output'/(name+'.png'));result['image']={'mode':im.mode,'size':list(im.size)}
 if 'A' in im.getbands():
  alpha=im.getchannel('A');hist=alpha.histogram();lo,hi=alpha.getextrema();result['image']['alpha_min']=lo;result['image']['alpha_max']=hi;result['image']['transparent_fraction']=sum(hist[:250])/(im.width*im.height)
 save(dest,result);return result

def stop(out):
 out.mkdir(parents=True,exist_ok=True)
 logs=subprocess.run(['docker','logs',NAME],capture_output=True,text=True);(out/'server.log').write_text(logs.stdout+logs.stderr)
 info=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True)
 if info.returncode==0:(out/'container.json').write_text(info.stdout)
 subprocess.run(['docker','stop','-t','10',NAME],capture_output=True);subprocess.run(['docker','rm',NAME],capture_output=True)

def wait_downloads():
 status('download','verify')
 while True:
  all_done=True
  for name in ['abenzerps','BennyDaBall','Comfy-Org']:
   p=ROOT/('download-'+name)/'download-status.json'
   if not p.exists():all_done=False;continue
   s=json.loads(p.read_text())
   if s['errors']:raise RuntimeError((name,s['errors']))
   all_done=all_done and s.get('complete',False)
  if all_done:return
  time.sleep(1)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profiles',nargs='*',default=['baseline','turbo-nvfp4','turbo-selective','turbo-bf16']);args=ap.parse_args()
 assert not command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').strip(),'GPU busy'
 fixtures()
 (ROOT/'monitor.stop').unlink(missing_ok=True)
 config=WORK/'util/talk/dist/sparktalk.yaml';digest=hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else None
 mon=subprocess.Popen([sys.executable,str(H/'monitor.py'),str(ROOT)])
 summary={};out=None
 try:
  profiles={'baseline':(BASE,False),'turbo-nvfp4':(selected_path('abenzerps'),True),'turbo-selective':(selected_path('BennyDaBall'),True),'turbo-bf16':(selected_path('Comfy-Org'),True)}
  for profile in args.profiles:
   if profile!='baseline':wait_downloads()
   out=setup(profile,*profiles[profile])
   request(out,'warmup',CASES[0][1],123)
   wait_downloads()
   rows=[]
   cases=CASES if profile!='turbo-bf16' else [c for c in CASES if c[0] in ('penguin','counting','korean','edit')]
   for name,prompt,seed in cases:rows.append(request(out,name,prompt,seed))
   if profile!='turbo-bf16':
    for i in range(3):request(out,'warm-'+str(i),CASES[0][1],20261010+i,cache=True)
   summary[profile]={'cases':len(rows),'complete':True};save(ROOT/'execution-summary.json',summary);stop(out);out=None
 except Exception as e:
  summary['error']=repr(e);save(ROOT/'execution-summary.json',summary);print('FAILED',repr(e),flush=True)
 finally:
  if out is not None:stop(out)
  else:
   if subprocess.run(['docker','inspect',NAME],capture_output=True).returncode==0:stop(ROOT/'failed-startup')
  (ROOT/'monitor.stop').touch();mon.wait(timeout=10);status('finished','done')
  save(ROOT/'config-preserved.json',{'unchanged':digest==hashlib.sha256(config.read_bytes()).hexdigest() if digest else None})

if __name__=='__main__':main()
