from pathlib import Path
import json,subprocess,time
r=Path(__file__).resolve().parent
while True:
 try:
  q=json.loads((r/'results/quality-controls.json').read_text())
  replay=json.loads((r/'results/language-replay-summary.json').read_text())
  if len(q)==2 and replay.get('completed'):
   assert all(x['passed'] for x in q),'Shortlist quality controls failed'
   break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(2)
def run(args):
 print('RUN',args,flush=True);subprocess.run(['python3',*map(str,args)],check=True)
for chunk in [8192]:
 run([r/'start_variant.py','--chunk',chunk,'--vocab','ko64k','--token',f'tune-ko64k-{chunk}-20260915'])
 paths=[]
 for mode in ['speed','prefill']:
  out=r/f'results/ko64k-{chunk}-{mode}.json';paths.append(out)
  run([r/'measure.py','--mode',mode,'--out',out])
 run([r/'summarize.py',*paths])
 for path in paths:
  d=json.loads(path.read_text())
  for row in d['rows']:
   if row.get('passed') is False:raise RuntimeError('Quality failure: '+row['name'])
print('CHUNKS_COMPLETE',flush=True)
(r/'results/chunks-complete.json').write_text(json.dumps({'completed':True,'time':time.time()}))
