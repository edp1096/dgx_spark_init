"""Wait for the selected guarded variant, then validate short and exact-1M input."""
import subprocess,time,json
from pathlib import Path
root=Path(__file__).resolve().parent
ready=root/'results/tune-ko64k-final1024-20260915/warmup.json'
for _ in range(1500):
 if ready.exists():break
 time.sleep(1)
else:raise TimeoutError('Selected model not ready')
subprocess.run(['python3',str(root/'stress.py')],check=True)
from measure import flush_cache
flush_cache()
source=root.parents[1]/'tp2'
subprocess.run(['docker','run','--rm','--name','qwen38-final-long-probe','--network','host','--memory','4g','--memory-swap','4g','--cpus','2','-e','HF_HUB_OFFLINE=1','-e','PYTHONUNBUFFERED=1','-v','/home/edp1096/.cache/huggingface:/hf:ro','-v',f'{source}:/probe:ro','-v',f'{root}/results:/results','--entrypoint','python3','dgx-sglang-qwen38-fn:sm121-tp2-vocab-v1','/probe/long_probe.py','--context','1048576','--output','/results/selected-1m.json'],check=True)
flush_cache()
(root/'results/final-validation-complete.json').write_text(json.dumps({'passed':True,'chunk':1024,'vocab':'ko64k'})+'\n')
