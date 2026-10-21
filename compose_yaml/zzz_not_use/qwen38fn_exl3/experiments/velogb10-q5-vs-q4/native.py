"""Same Q5 weights and 1M Q8 profile on the existing native ExLlama/Tabby image."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('q5_runner',H/'run.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);a=q.a
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root;r.mkdir(mode=0o700,exist_ok=True);out=r/'q5-1m';out.mkdir(exist_ok=True);a.ROOT=r
cfg=q.WORKSPACE/'util/talk/dist/sparktalk.yaml';before=hashlib.sha256(cfg.read_bytes()).hexdigest()
assert not a.command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').strip()
name='exl3-q5-native-audit';a.AUX=q.AUX
initial=json.loads(a.command('docker','inspect',*q.AUX));assert all(not d['State']['Running'] for d in initial)
manifest={'runtime':'ExLlamav3 1.5.4 + TabbyAPI native existing image','context_tokens':1048576,'kv':'Q8','mtp_steps':3,'ngram':'K6 SSD streaming','vision':True,'production_config_sha256_before':before,'started':time.time()}
(r/'manifest.json').write_text(json.dumps(manifest,indent=2));(r/'phase').write_text('q5-native-startup')
mon=subprocess.Popen([sys.executable,str(H/'monitor.py'),str(r),'native-exllamav3']);results={}
try:
 a.command('docker','start','sparktalk-nemotron-asr');a.wait('http://127.0.0.1:8693','/ready')
 cmd=['docker','run','-d','--name',name,'--gpus','all','--memory','116g','--memory-swap','116g','--cpuset-cpus','5-9,15-19','--ipc','host','--network','bridge','-p','127.0.0.1:19336:30000','-v',str(Path.home()/'.cache/huggingface')+':/hf:ro','-v',str(r/'cache')+':/runtime','-v',str(r/'launch.py')+':/opt/sparktalk-qwen38fn_exl3/launch.py:ro']
 for key,value in {'EXL3_INT8_GEMV':'0','EXL3_NGRAM_STREAM':'1','EXL3_MOE_CPU_OFFLOAD':'0','EXL3_MOE_CPU_SPLIT':'0','HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','TRITON_CACHE_DIR':'/runtime/triton'}.items():cmd+=['-e',key+'='+value]
 cmd+=['--entrypoint','python','sparktalk-qwen38fn_exl3:1.5.4-managed1','/opt/sparktalk-qwen38fn_exl3/launch.py'];(r/'command.json').write_text(json.dumps(cmd,indent=2));a.command(*cmd)
 pid=a.command('docker','inspect','--format','{{.State.Pid}}',name);(r/'velo.pid').write_text(pid)
 deadline=time.monotonic()+900
 while True:
  try:card=a.http('http://127.0.0.1:19336','/v1/model',timeout=2);break
  except (OSError,ValueError):
   state=json.loads(a.command('docker','inspect',name))[0]['State']
   if not state['Running']:raise RuntimeError('Native runtime exited; see server.log')
   if (r/'memory-floor.json').exists():raise RuntimeError('Memory floor reached')
   if time.monotonic()>deadline:raise TimeoutError('Native startup')
   time.sleep(1)
 assert card['id']=='q5-native-audit' and card['parameters']['max_seq_len']==1048576 and card['parameters']['cache_size']==1048576 and card['parameters']['cache_mode']=='Q8' and card['parameters']['use_vision'],card
 (out/'model-card.json').write_text(json.dumps(card,indent=2))
 code="""import os
from pathlib import Path
files=sorted(Path('/runtime/models/q5-native-audit').glob('model-*-of-00011.safetensors'))
assert len(files)==11
resolved={str(p.resolve()) for p in files}
maps=Path('/proc/1/maps').read_text();fds={str(p.resolve()) for p in Path('/proc/1/fd').iterdir()}
assert not resolved.intersection(fds) and not any(p in maps for p in resolved)
for p in files:
 with p.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
print('closed main-weight pages advised; n-gram untouched')"""
 a.command('docker','exec',name,'python','-c',code)
 sanity=a.stream('http://127.0.0.1:19336','q5-native-audit',[{'role':'user','content':'17×23의 결과를 숫자만 답해라.'}],32);(out/'sanity.json').write_text(json.dumps(sanity,ensure_ascii=False,indent=2));assert sanity['text'].strip()=='391',sanity
 q.install_checked_auxiliary_helpers(r)
 results=a.tests(out,'q5-native-1m','http://127.0.0.1:19336','q5-native-audit');manifest['status']='complete';manifest['resident_auxiliaries_verified']=True
 for required in ('schema','coding','vision-quality','recall-1m-run','vision-long','post-stress-chat'):
  row=json.loads((out/(required+'.json')).read_text());value=row.get('result',{})
  if not row['ok'] or value.get('pass') is False or (required in ('coding','vision-quality') and value.get('passed')!=value.get('total')):manifest.setdefault('failed_checks',[]).append(required)
 if manifest.get('failed_checks'):manifest['status']='validation-failed'
except Exception as e:
 manifest['status']='memory-floor' if (r/'memory-floor.json').exists() else 'failed';manifest['error']=repr(e);print('NATIVE FAILED',repr(e),flush=True)
finally:
 (r/'phase').write_text('q5-native-cleanup')
 logs=subprocess.run(['docker','logs',name],capture_output=True,text=True);(out/'server.log').write_text(logs.stdout+logs.stderr)
 info=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
 if info.returncode==0:(r/'container.json').write_text(info.stdout)
 subprocess.run(['docker','stop','-t','10',name,*q.AUX],capture_output=True)
 subprocess.run(['docker','rm',name],capture_output=True)
 manifest['production_config_unchanged']=before==hashlib.sha256(cfg.read_bytes()).hexdigest();manifest['finished']=time.time();manifest['results']=results
 (r/'manifest.json').write_text(json.dumps(manifest,indent=2));(r/'monitor.stop').touch();mon.wait(timeout=10);(r/'phase').write_text('q5-native-finished')
print(json.dumps(manifest,indent=2),flush=True)
