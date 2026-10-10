"""Targeted editing checks and the existing 3bit/1M resident bundle footprint."""
import hashlib,json,subprocess,sys,time,urllib.request
from PIL import Image
import run as a

OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
OWN=['sparktalk-qwen38fn_exl3','sparktalk-nemotron-asr','sparktalk-qwen3-tts','sparktalk-embedding']
def http(port,path,data=None,raw=False,timeout=180):
 req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with OPENER.open(req,timeout=timeout) as r:return r.read() if raw else json.load(r)
def wait(port,path):
 end=time.monotonic()+600
 while time.monotonic()<end:
  try:return http(port,path,timeout=3)
  except (OSError,ValueError):time.sleep(1)
 raise TimeoutError((port,path))
def identities():
 return {d['Name'].lstrip('/'):{'id':d['Id'],'pid':d['State']['Pid'],'running':d['State']['Running'],'oom':d['State']['OOMKilled']} for d in json.loads(a.command('docker','inspect',*OWN))}
def main():
 assert not a.command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').strip(),'GPU busy'
 initial=identities();assert not any(d['running'] for d in initial.values()),initial
 a.save(a.ROOT/'joint-initial-identities.json',initial)
 cfg=a.WORK/'util/talk/dist/sparktalk.yaml';digest=hashlib.sha256(cfg.read_bytes()).hexdigest()
 (a.ROOT/'monitor.stop').unlink(missing_ok=True)
 assert not (a.ROOT/'memory-floor.json').exists()
 mon=subprocess.Popen([sys.executable,str(a.H/'monitor.py'),str(a.ROOT)])
 out=None;report={};started=[]
 try:
  variants=[('baseline',a.BASE,False),('turbo-nvfp4',a.selected_path('abenzerps'),True),('turbo-selective',a.selected_path('BennyDaBall'),True),('turbo-bf16',a.selected_path('Comfy-Org'),True)]
  for name,model,turbo in variants:
   out=a.setup('editing-check-'+name,model,turbo)
   a.request(out,'warmup',a.CASES[0][1],123)
   if name=='turbo-bf16':a.request(out,'multiref',a.CASES[-1][1],20261010)
   a.request(out,'multiref-seed2',a.CASES[-1][1],42)
   a.stop(out);out=None
  a.status('joint-services','startup')
  for name,port,path in [(OWN[0],18002,'/v1/model'),(OWN[1],8693,'/ready'),(OWN[2],8692,'/ready'),(OWN[3],8701,'/health')]:
   a.command('docker','start',name);started.append(name);value=wait(port,path);a.save(a.ROOT/(name+'-joint-ready.json'),value)
   if port==18002:
    p=value['parameters'];assert p['max_seq_len']==p['cache_size']==1048576 and p['cache_mode']=='Q8' and p['use_vision'],value
    a.command('docker','exec',name,'python','/opt/sparktalk-qwen38fn_exl3/release_weight_cache.py')
  llm={'model':'qwen38fn_exl3','messages':[{'role':'user','content':'17×23의 결과를 숫자만 답해라.'}],'max_tokens':32,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
  a.save(a.ROOT/'joint-chat-before.json',http(18002,'/v1/chat/completions',llm))
  embedding=http(8701,'/v1/encode',{'task':'query','inputs':[{'text':'메모리에 상주하는 모델의 사용량은?'}]})
  a.save(a.ROOT/'joint-embedding-check.json',{'dimensions':len(embedding['data'][0]['embedding'])})
  speech=http(8692,'/v1/audio/speech',{'model':'qwen3-tts-0.6b-q8','input':'안녕하세요. 이미지 생성 속도를 확인합니다.','voice':'sohee','language':'Korean','speed':1,'response_format':'wav'},raw=True)
  assert len(speech)>1000;(a.ROOT/'joint-speech.wav').write_bytes(speech)
  before=identities();a.save(a.ROOT/'joint-identities-before.json',before)
  Image.open(a.ROOT/'baseline/output/hands.png').convert('RGB').save(a.ROOT/'inputs/hands-reference.png')
  prompt='Edit this photograph: change ONLY the woman\'s cream blouse to a dark navy blue blouse. Preserve the same woman\'s identity, age, facial features, hair, pose, both hands, transparent glass cup of tea, and background. Photorealistic. No other changes.'
  for name,model,turbo in variants[:3]:
   out=a.setup('joint-'+name,model,turbo,joint=True)
   a.request(out,'warmup',a.CASES[0][1],123)
   rows=[a.request(out,'penguin',a.CASES[0][1],20261010),a.request(out,'photo-edit',prompt,20261010)]
   report[name]={'results':rows,'identities_unchanged':identities()==before}
   a.save(a.ROOT/'joint-summary.json',report);a.stop(out);out=None
  a.save(a.ROOT/'joint-chat-after.json',http(18002,'/v1/chat/completions',llm))
  a.save(a.ROOT/'joint-identities-after.json',identities())
 except Exception as e:
  report['error']=repr(e);a.save(a.ROOT/'joint-summary.json',report);print('FAILED',repr(e),flush=True)
 finally:
  if out is not None:a.stop(out)
  elif subprocess.run(['docker','inspect',a.NAME],capture_output=True).returncode==0:a.stop(a.ROOT/'followup-failed-startup')
  for name in reversed(started):subprocess.run(['docker','stop','-t','10',name],capture_output=True)
  (a.ROOT/'monitor.stop').touch();mon.wait(timeout=10);a.status('finished','done')
  a.save(a.ROOT/'followup-cleanup.json',{'config_unchanged':digest==hashlib.sha256(cfg.read_bytes()).hexdigest(),'final_identities':identities(),'initial_identities':initial})
if __name__=='__main__':main()
