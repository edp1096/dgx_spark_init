"""Repeat ordinary generation after all schema/concurrency tests, with no competing requests."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from compare import stream

HERE=Path(__file__).resolve().parent
BASELINE=Path('/home/edp1096/.cache/model-download-jobs/velogb10-exllama-comparison-20261009/velo-full-vocab')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    deadline=time.monotonic()+2400
    while not (a.root/'completed.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Schema qualification did not finish')
        time.sleep(2)
    out=a.root/'clean-speed'
    with (a.root/'clean-speed-launcher.log').open('w') as log:
        p=subprocess.Popen([sys.executable,str(HERE/'trial.py'),'--root',str(a.root),'--label',out.name,
            '--context','1048576','--factor','4','--port','19322','--max-batch','1','--prefix-cache','on',
            '--prefix-ckpt-mem-gb','4','--cpuset','5-9,15-19','--mtp-head-n','0','--pdl','0'],stdout=log,stderr=subprocess.STDOUT)
        try:
            while not (out/'short-complete').exists():
                if p.poll() is not None:raise RuntimeError('Clean speed server failed')
                time.sleep(2)
            results=[]
            for name in ['warmup_0','warmup_1','korean_0','code_0','korean_1','code_1','korean_2','code_2']:
                b=json.loads((BASELINE/(name+'.request.json')).read_text());messages=b.pop('messages');budget=b.pop('max_tokens')
                r=stream('http://127.0.0.1:19322','velo-yarn-audit',messages,budget,**b);r['name']=name
                results.append(r);(out/'speed.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
                print(name,round(r['ttft'],3),round(r['decode_tps'],2),flush=True)
        finally:
            (out/'stop').touch();p.wait(timeout=40)


if __name__=='__main__':main()
