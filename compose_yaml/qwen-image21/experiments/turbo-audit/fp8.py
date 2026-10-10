"""Test FP8 after the NVFP4 reference-edit regression, without overlapping runs."""
import hashlib,json,os,subprocess,sys,time
import run as a
import followup as b

def main():
 while not (a.ROOT/'followup-cleanup.json').exists():time.sleep(1)
 assert not a.command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').strip(),'GPU busy'
 env=os.environ.copy();env.update(EXL3_DOWNLOAD_ROOT=str(a.ROOT/'download-fp8'),EXL3_DOWNLOAD_METADATA='metadata.json',EXL3_DOWNLOAD_FILES='1',EXL3_DOWNLOAD_STREAMS='12')
 downloader=a.WORK/'compose_yaml/qwen38fn_exl3/experiments/velogb10-q4-vs-nvfp4/download.py'
 a.status('download-fp8','verify');subprocess.run([sys.executable,str(downloader)],env=env,check=True)
 initial=b.identities();assert not any(d['running'] for d in initial.values()),initial
 cfg=a.WORK/'util/talk/dist/sparktalk.yaml';digest=hashlib.sha256(cfg.read_bytes()).hexdigest()
 (a.ROOT/'monitor.stop').unlink(missing_ok=True);assert not (a.ROOT/'memory-floor.json').exists()
 mon=subprocess.Popen([sys.executable,str(a.H/'monitor.py'),str(a.ROOT)])
 out=None;started=[];report={};model=a.selected_path('fp8')
 try:
  out=a.setup('turbo-fp8',model,True);a.request(out,'warmup',a.CASES[0][1],123)
  for name,prompt,seed in a.CASES:a.request(out,name,prompt,seed)
  a.request(out,'multiref-seed2',a.CASES[-1][1],42)
  for i in range(3):a.request(out,'warm-'+str(i),a.CASES[0][1],20261010+i,cache=True)
  report['isolated_complete']=True;a.stop(out);out=None
  # Keep the same 1M Q8-KV profile and all auxiliary model processes resident.
  a.status('fp8-joint-services','startup')
  for name,port,path in [(b.OWN[0],18002,'/v1/model'),(b.OWN[1],8693,'/ready'),(b.OWN[2],8692,'/ready'),(b.OWN[3],8701,'/health')]:
   a.command('docker','start',name);started.append(name);value=b.wait(port,path)
   a.save(a.ROOT/(name+'-fp8-joint-ready.json'),value)
   if port==18002:
    p=value['parameters'];assert p['max_seq_len']==p['cache_size']==1048576 and p['cache_mode']=='Q8' and p['use_vision'],value
    a.command('docker','exec',name,'python','/opt/sparktalk-qwen38fn_exl3/release_weight_cache.py')
  llm={'model':'qwen38fn_exl3','messages':[{'role':'user','content':'17×23의 결과를 숫자만 답해라.'}],'max_tokens':32,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
  a.save(a.ROOT/'fp8-joint-chat-before.json',b.http(18002,'/v1/chat/completions',llm))
  embedding=b.http(8701,'/v1/encode',{'task':'query','inputs':[{'text':'메모리에 상주하는 모델의 사용량은?'}]})
  assert len(embedding['data'][0]['embedding'])==768
  speech=b.http(8692,'/v1/audio/speech',{'model':'qwen3-tts-0.6b-q8','input':'안녕하세요. 이미지 생성 속도를 확인합니다.','voice':'sohee','language':'Korean','speed':1,'response_format':'wav'},raw=True)
  assert len(speech)>1000
  before=b.identities();a.save(a.ROOT/'fp8-joint-identities-before.json',before)
  out=a.setup('joint-turbo-fp8',model,True,joint=True)
  a.request(out,'warmup',a.CASES[0][1],123)
  prompt='Edit this photograph: change ONLY the woman\'s cream blouse to a dark navy blue blouse. Preserve the same woman\'s identity, age, facial features, hair, pose, both hands, transparent glass cup of tea, and background. Photorealistic. No other changes.'
  report['joint_results']=[a.request(out,'penguin',a.CASES[0][1],20261010),a.request(out,'photo-edit',prompt,20261010)]
  report['identities_unchanged']=b.identities()==before;a.save(a.ROOT/'fp8-joint-identities-after.json',b.identities())
  a.stop(out);out=None
 except Exception as e:
  report['error']=repr(e);print('FAILED',repr(e),flush=True)
 finally:
  a.save(a.ROOT/'fp8-execution-summary.json',report)
  if out is not None:a.stop(out)
  elif subprocess.run(['docker','inspect',a.NAME],capture_output=True).returncode==0:a.stop(a.ROOT/'fp8-failed-startup')
  for name in reversed(started):subprocess.run(['docker','stop','-t','10',name],capture_output=True)
  (a.ROOT/'monitor.stop').touch();mon.wait(timeout=10);a.status('finished','done')
  a.save(a.ROOT/'fp8-cleanup.json',{'config_unchanged':digest==hashlib.sha256(cfg.read_bytes()).hexdigest(),'final_identities':b.identities(),'initial_identities':initial})
if __name__=='__main__':main()
