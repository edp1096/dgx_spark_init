from pathlib import Path
import subprocess,sys,json,time
r=Path(__file__).parent
cmd=['docker','run','--rm','--name','gemma31-high-mtp-kernel-check','--gpus','all','--ipc','host','--network','none','-v',str(r/'verify-work')+':/check','--entrypoint','python3','gemma31-assistant-bpw:trial','/check/test-high-width.py']
(r/'kernel-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (r/'kernel-check.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
assert (r/'verify-work/HIGH_WIDTH_COMPLETE').exists()
print('KERNEL_WIDTHS_19_22_25_PASSED',flush=True)
with (r/'sweep.log').open('w') as log:
 subprocess.run([sys.executable,str(r/'run-high-mtp.py'),'splitkv-b4-high-9','splitkv-b4-high-18','splitkv-b4-high-21','splitkv-b4-high-24'],stdout=log,stderr=subprocess.STDOUT,check=True)
(r/'ALL_COMPARISONS_FINISHED').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z')+'\n')
print('ALL_HIGH_MTP_COMPARISONS_FINISHED',flush=True)
