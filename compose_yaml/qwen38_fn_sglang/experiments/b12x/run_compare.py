import json,os,sys,time,subprocess,urllib.request,datetime,shlex,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'));import manage_tp2 as m
os.environ.update(QWEN_TP2_API_PORT='8014',QWEN_TP2_DIST_PORT='29980',QWEN_TP2_BIND='127.0.0.1',QWEN_TP2_MODEL='qwen38-sglang-b12x-probe')
base='http://127.0.0.1:8014'
parser=argparse.ArgumentParser();parser.add_argument('--variant',action='append',choices=['cutlass','moe','head']);parser.add_argument('--context',type=int,choices=[262144,524288,1048576],default=262144);parser.add_argument('--long',action='store_true');args=parser.parse_args()
for variant in args.variant or ['cutlass','head']:
 enabled='1' if variant=='moe' else '0'
 token='sgl-b12x-'+variant+'-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S');out=ROOT/'results'/token;out.mkdir(parents=True)
 os.environ['SGLANG_B12X_PROBE_MOE']=enabled
 os.environ['SGLANG_B12X_PROBE_HEAD']='1' if variant=='head' else '0'
 provenance={'variant':variant,'context':args.context,'long':args.long,'image':m.IMAGE,'moe':enabled,'head':os.environ['SGLANG_B12X_PROBE_HEAD'],'source_hashes':{}}
 for rel in ['adapter.py','bench.py','runtime/tp2/entrypoint.py','runtime/tp2/adapter.py','runtime/compose.tp2.yaml']:
  data=(ROOT/rel).read_bytes();provenance['source_hashes'][rel]=hashlib.sha256(data).hexdigest()
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
 proc=None
 try:
  m.start(args.context,token);deadline=time.monotonic()+1800
  def check():
   for rank in (0,1):
    d=m.inspect(rank,m.name(rank));assert d and d['State']['Running'],(rank,d['State'] if d else None)
    code=f"import os,json;from pathlib import Path;p=Path.home()/'.local/state/qwen38-tp2'/{token!r};assert not (p/'tripped-rank{rank}.json').exists();os.kill(json.loads((p/'ready-rank{rank}.json').read_text())['pid'],0)"
    m.run(rank,['python3','-c',code],capture_output=True)
  while True:
   check()
   try:
    with urllib.request.urlopen(base+'/health',timeout=3) as f:assert f.status==200
    break
   except Exception:
    if time.monotonic()>deadline:raise TimeoutError('Startup timeout')
    print(variant,'loading',flush=True);time.sleep(20)
  print(variant,'ready, benchmarking',flush=True)
  cmd=['docker','run','--rm','--name','sglang-b12x-client','--network','host','--memory','3g','--memory-swap','3g','--cpus','2','-e','HF_HUB_OFFLINE=1','-e','PYTHONUNBUFFERED=1','-v',str(Path.home()/'.cache/huggingface')+':/hf:ro','-v',str(ROOT)+':/probe:ro','-v',str(out)+':/results','--entrypoint','python3',m.IMAGE,'/probe/bench.py','--output','/results/bench.json']
  if args.long:
   cmd=cmd[:cmd.index('/probe/bench.py')]+['/probe/long_probe.py','--context',str(args.context),'--url',base,'--output','/results/retrieval.json']
  with (out/'bench.log').open('w') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);deadline=time.monotonic()+(7200 if args.long else 1800)
   while proc.poll() is None:
    check()
    if time.monotonic()>deadline:raise TimeoutError('Bench timeout')
    print(variant,'benchmark running',flush=True);time.sleep(20)
  assert proc.returncode==0,(out/'bench.log').read_text()[-3000:]
  print(variant,json.loads((out/('retrieval.json' if args.long else 'bench.json')).read_text()),flush=True)
 except BaseException as e:
  (out/'error.txt').write_text(repr(e));raise
 finally:
  if proc and proc.poll() is None:subprocess.run(['docker','stop','-t','2','sglang-b12x-client'],capture_output=True)
  m.stop()
  for rank in (0,1):
   d=m.inspect(rank,m.name(rank));(out/f'state-rank{rank}.json').write_text(json.dumps(d['State'] if d else None,indent=2))
   r=m.run(rank,['docker','logs',m.name(rank)],capture_output=True,text=True);(out/f'server-rank{rank}.log').write_text(r.stdout+r.stderr)
   src=str(Path.home()/'.local/state/qwen38-tp2'/token)+'/'
   if rank:src=m.WORKER+':'+src
   subprocess.run(['rsync','-a',src,str(out/f'guard-rank{rank}')+'/'],check=True)
