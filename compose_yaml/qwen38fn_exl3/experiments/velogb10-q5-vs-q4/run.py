"""Q5 on the same qualified Q4 engine, 1M KV and resident ASR/embedding profile."""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'velogb10-q4-vs-nvfp4'
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('matched_q4_trial',OLD/'run.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
SOURCE=a.SOURCE;WORKSPACE=a.WORKSPACE
AUX=['sparktalk-nemotron-asr','sparktalk-embedding','sparktalk-extra-media']

def advise_closed_weights(model,pid):
 files=sorted(model.glob('model-*-of-00011.safetensors'));assert len(files)==11
 resolved={str(p.resolve()) for p in files};active=set()
 for process in Path('/proc').glob('[0-9]*'):
  try:
   maps=(process/'maps').read_text();active.update(p for p in resolved if p in maps)
   for fd in (process/'fd').iterdir():
    try:
     v=str(fd.resolve(strict=True))
     if v in resolved:active.add(v)
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 assert not active,active
 for p in files:
  with p.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 return {'files':len(files),'ple_untouched':True,'policy':'closed, unmapped checkpoint files only'}

def install_checked_auxiliary_helpers(root):
 original_record=a.record
 def record(out,name,fn):
  result=original_record(out,name,fn)
  mandatory={'sparktalk-nemotron-asr-health','sparktalk-embedding-health','embedding','embedding-memory','diarization','hybrid-retrieval'}
  if name in mandatory and not result['ok']:raise RuntimeError(f'Resident auxiliary {name} failed: {result}')
  return result
 a.record=record
 def wait(base,path='/health',process=None):
  deadline=time.monotonic()+(1500 if process is not None else 180)
  name={'http://127.0.0.1:8693':'sparktalk-nemotron-asr','http://127.0.0.1:8701':'sparktalk-embedding','http://127.0.0.1:8690':'sparktalk-extra-media'}.get(base)
  initial_restarts=json.loads(a.command('docker','inspect',name))[0]['RestartCount'] if name else 0
  last=0
  while time.monotonic()<deadline:
   if (root/'memory-floor.json').exists():raise RuntimeError('Memory floor reached')
   if process is not None and process.poll() is not None:raise RuntimeError('Candidate exited; see server.log')
   try:return a.http(base,path,timeout=2)
   except (OSError,ValueError):pass
   if name and time.monotonic()-last>3:
    data=json.loads(a.command('docker','inspect',name))[0]
    if not data['State']['Running'] or data['RestartCount']>=initial_restarts+3:
     (root/(name+'-failure.log')).write_text(a.command('docker','logs',name))
     raise RuntimeError(f'{name} exited/restarted during initialization: {data["State"]["ExitCode"]}')
    last=time.monotonic()
   time.sleep(1)
  raise TimeoutError(base+path)
 a.wait=wait

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--model-only',action='store_true');ap.add_argument('--prefix-ckpt-gib',type=int,default=4);ap.add_argument('--prefill-chunk',type=int,default=2048);ap.add_argument('--memory-gib',type=int,default=112);ap.add_argument('--mem-watchdog-gb',type=float,default=5.0);ap.add_argument('--prewarm-asr',action='store_true');args=ap.parse_args()
 root=args.root;a.ROOT=root;a.AUX=AUX
 install_checked_auxiliary_helpers(root)
 a.phase('waiting-for-verified-q5-download')
 while True:
  try:status=json.loads((root/'download-status.json').read_text())
  except FileNotFoundError:time.sleep(1);continue
  if status.get('errors'):raise RuntimeError(status['errors'])
  if status.get('complete'):break
  time.sleep(1)
 meta=json.loads((root/'remote-metadata.json').read_text());model=Path(status['snapshot'])
 cfg=WORKSPACE/'util/talk/dist/sparktalk.yaml';before=hashlib.sha256(cfg.read_bytes()).hexdigest()
 initial=json.loads(a.command('docker','inspect',*AUX))
 assert all(not d['State']['Running'] for d in initial),'Auxiliary model services must be idle before the isolated trial'
 compute=a.command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits')
 assert not compute.strip(),'GPU is in use; refusing to interfere with another model: '+compute
 out=root/'q5-1m';out.mkdir(exist_ok=True)
 binary=SOURCE/'target/release/gb10_inference'
 command=['taskset','-c','5-9,15-19',str(binary),'--server','--model-dir',str(model),'--host','127.0.0.1','--port','19335','--model-name','q5-exl3-audit','--max-seq-len','1048576','--rope-yarn-factor','4','--max-batch','1','--ple-ram','ssd','--kv-cache','q8','--exl3-mtp-k','3','--draft-confidence','0','--prefix-cache','on','--prefix-ckpt-mem-gb',str(args.prefix_ckpt_gib),'--prefill-chunk',str(args.prefill_chunk),'--tune-table','off','--reasoning-effort','none','--thinking','off','--exl3-pdl','0','--exl3-mtp-head-n','0','--mem-watchdog-gb',str(args.mem_watchdog_gb)]
 manifest={'started':time.time(),'repo':meta['id'],'revision':meta['sha'],'context_tokens':1048576,'kv':'q8','yarn_factor':4,'mtp_steps':3,'prefix_ckpt_gib':args.prefix_ckpt_gib,'prefill_chunk':args.prefill_chunk,'cgroup_memory_gib':args.memory_gib,'engine_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'asr_diarization_embedding_resident':not args.model_only,'prewarm_asr':args.prewarm_asr,'internal_mem_watchdog_gb':args.mem_watchdog_gb,'no_image_generation_or_tts':True,'q4_baseline':str(a.WORKSPACE/'compose_yaml/qwen38fn_exl3/experiments/velogb10-q4-vs-nvfp4/results.json'),'production_config_sha256_before':before,'main_weight_gib':sum(e.get('size',0) for e in meta['siblings'] if e['rfilename'].startswith('model-'))/2**30}
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'command.json').write_text(json.dumps(command,indent=2))
 monitor=subprocess.Popen([sys.executable,str(HERE/'monitor.py'),str(root)])
 velo=None;log=None;result={}
 try:
  if args.prewarm_asr:
   a.phase('q5-asr-cold-cuda-prewarm')
   a.command('docker','start','sparktalk-nemotron-asr')
   a.wait('http://127.0.0.1:8693','/ready')
  a.phase('q5-1m-startup')
  cmd=['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit=velo-q5-comparison','-p','MemoryMax='+str(args.memory_gib)+'G','-p','MemorySwapMax=0','-p','KillSignal=SIGKILL','-p','WorkingDirectory='+str(SOURCE),'-p','Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu',*command]
  log=(out/'server.log').open('w');velo=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  for _ in range(100):
   pid=a.command('systemctl','--user','show','velo-q5-comparison','-p','MainPID','--value')
   if pid.isdigit() and int(pid):break
   time.sleep(.2)
  else:raise RuntimeError('Velo unit did not start')
  (root/'velo.pid').write_text(pid);a.wait('http://127.0.0.1:19335',process=velo)
  card=a.http('http://127.0.0.1:19335','/v1/models');assert card['data'][0]['max_model_len']==1048576
  (out/'model-card.json').write_text(json.dumps(card,indent=2))
  a.record(out,'closed-cache',lambda:advise_closed_weights(model,int(pid)))
  try:
   a.phase('q5-resident-auxiliary-validation')
   if not args.model_only:a.auxiliary(out)
   manifest['resident_auxiliaries_verified']=not args.model_only
  except Exception as err:
   manifest['resident_auxiliaries_verified']=False
   manifest['resident_auxiliary_error']=repr(err)
   subprocess.run(['docker','stop','-t','10',*AUX],capture_output=True)
   if (root/'memory-floor.json').exists():raise
   a.wait('http://127.0.0.1:19335',process=velo)
   print('Resident profile failed; qualifying model-only behavior separately',repr(err),flush=True)
  a.auxiliary=lambda output:None
  (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
  result=a.tests(out,'q5-1m','http://127.0.0.1:19335','q5-exl3-audit')
  a.wait('http://127.0.0.1:19335',process=velo)
  required=['schema','coding','vision-quality','recall-1m-run','vision-long','post-stress-chat']
  failed=[]
  for name in required:
   row=json.loads((out/(name+'.json')).read_text())
   value=row.get('result',{})
   passed=row['ok'] and value.get('pass',True)
   if name in ('coding','vision-quality'):passed=passed and value.get('passed')==value.get('total')
   if name=='post-stress-chat':passed=passed and value.get('text','').strip()=='391'
   if not passed:failed.append(name)
  manifest['failed_required_checks']=failed
  manifest['status']='validation-failed' if failed else ('complete-model-only' if args.model_only else ('complete' if manifest['resident_auxiliaries_verified'] else 'incompatible-resident-profile'))
 except Exception as e:
  manifest['status']='memory-floor' if (root/'memory-floor.json').exists() else 'failed'
  manifest['error']=repr(e)
  print('TRIAL FAILED',repr(e),flush=True)
 finally:
  (root/'phase').write_text('q5-cleanup')
  subprocess.run(['docker','stop','-t','10',*AUX],capture_output=True)
  subprocess.run(['systemctl','--user','stop','velo-q5-comparison'],capture_output=True)
  if velo is not None:
   try:velo.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(velo.pid,signal.SIGKILL)
  if log:log.close()
  manifest['production_config_unchanged']=before==hashlib.sha256(cfg.read_bytes()).hexdigest()
  (root/'monitor.stop').touch();monitor.wait(timeout=10)
  manifest['finished']=time.time();manifest['results']=result
  (root/'manifest.json').write_text(json.dumps(manifest,indent=2));(root/'phase').write_text('q5-finished')
 print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
