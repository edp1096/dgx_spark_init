import pathlib,json,subprocess,sys,time,urllib.request
r=pathlib.Path(__file__).parent
kind=sys.argv[1]
assert kind in ['stock','korean','hybrid-stock','hybrid-korean']
name='qwen38-b12x-hybrid-20261020'
exists=subprocess.run(['docker','inspect',name],capture_output=True).returncode==0
if exists:
 subprocess.run(['docker','stop','-t','10',name],check=True,stdout=subprocess.DEVNULL)
 stamp=time.strftime('%H%M%S',time.gmtime())
 with (r/('server-'+stamp+'.log')).open('w') as f:subprocess.run(['docker','logs',name],stdout=f,stderr=subprocess.STDOUT)
 subprocess.run(['docker','rename',name,name+'-'+stamp],check=True)
for _ in range(30):
 available=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 if available>108*2**30:break
 time.sleep(1)
else:raise RuntimeError('Free the model-serving GPUs before launching the TP1 experiment; need 108 GiB MemAvailable')
config=json.loads((r/'tensorfold-launch.json').read_text());cmd=config['command']
binary='/bridge/install-ko/native/bin/tensorfold-native' if 'korean' in kind else '/bridge/install-stock/native/bin/tensorfold-native'
cmd[cmd.index('--entrypoint')+1]=binary
if '--cache-off' in sys.argv:cmd+=['--prompt-cache-gib','0']
if '--fp32-router' in sys.argv:
 i=cmd.index('local/qwen38-fp4-local:csf-yarn-20261016');cmd[i:i]=['-e','B12X_W4A16_FP32_TOPK_WEIGHTS=1']
if not kind.startswith('hybrid'):
 i=next(i for i,x in enumerate(cmd) if x.startswith('TF_B12X_BRIDGE='));del cmd[i-1:i+1]
subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
start=time.monotonic()
for _ in range(1800):
 try:
  with urllib.request.urlopen('http://127.0.0.1:18018/health',timeout=.5) as f:
   if f.status==200:break
 except Exception:pass
 if subprocess.run(['docker','inspect','--format','{{.State.Running}}',name],capture_output=True,text=True).stdout.strip()!='true':raise RuntimeError('Trial exited')
 time.sleep(1)
else:raise RuntimeError('Readiness timeout')
print(json.dumps({'trial':kind,'ready_s':time.monotonic()-start}),flush=True)
(r/'active-trial.json').write_text(json.dumps({'trial':kind,'command':cmd,'ready_s':time.monotonic()-start},indent=2))
