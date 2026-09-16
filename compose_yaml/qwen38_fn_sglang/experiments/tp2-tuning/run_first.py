from pathlib import Path
import json,subprocess,time
r=Path(__file__).resolve().parent
while not json.loads((r/'results/language-comparison.json').read_text()).get('completed'):time.sleep(2)
def run(args):
 print('RUN',args,flush=True);subprocess.run(['python3',*map(str,args)],check=True)
run([r/'measure.py','--mode','language-strict','--out',r/'results/language-strict.json'])
run([r/'language_controls.py'])
run([r/'measure.py','--mode','speed','--out',r/'results/baseline-speed-clean.json'])
run([r/'measure.py','--mode','prefill','--out',r/'results/baseline-prefill-clean.json'])
run([r/'start_variant.py','--chunk','1024','--vocab','ko64k','--token','tune-ko64k-1024-20260915'])
run([r/'measure.py','--mode','speed','--out',r/'results/ko64k-1024-speed.json'])
run([r/'measure.py','--mode','prefill','--out',r/'results/ko64k-1024-prefill.json'])
print('FIRST_COMPARISON_COMPLETE',flush=True)
