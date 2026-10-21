import pathlib,json,subprocess,os,sys,shutil
r=pathlib.Path(__file__).parent
with (r/'hybrid-nocache-launch.log').open('w') as f:subprocess.run([sys.executable,str(r/'trial-launch.py'),'hybrid-korean','--fp32-router','--cache-off'],stdout=f,stderr=subprocess.STDOUT,check=True)
with (r/'hybrid-fp32-nocache-exactness.log').open('w') as f:subprocess.run([sys.executable,str(r/'check-exactness.py'),'hybrid-fp32-nocache'],stdout=f,stderr=subprocess.STDOUT,check=True)
checks=json.loads((r/'hybrid-fp32-nocache-exactness.json').read_text())
if not all(x['drafted_equal_plain'] and x['plain_repeat_equal'] for x in checks):
 print('No-cache hybrid exactness still failed; speed screen is disqualified',flush=True);sys.exit(1)
env=dict(os.environ);env['BENCH_CASE_IDS']=','.join(['warmup-off','warmup-medium']+[f'tg-explain-{mode}-{rep}' for rep in [0,1] for mode in ['off','medium']])
with (r/'hybrid-fp32-nocache-bench.log').open('w') as f:subprocess.run([sys.executable,str(r/'bench.py'),'tensorfold'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
shutil.copy(r/'tensorfold-rows.json',r/'hybrid-fp32-nocache-rows.json')
print('No-cache hybrid TG screen completed',flush=True)
