"""After the main A/B stops, isolate Velo's draft-vocabulary truncation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--resume', action='store_true', help='Attach to the already running isolated full-vocabulary trial')
    a = ap.parse_args()
    if a.resume:
        out = a.root / 'velo-full-vocab'
        try:
            deadline = time.monotonic() + 600
            while not (out / 'suite-summary.json').exists():
                if (out / 'exit.json').exists() or time.monotonic() > deadline:
                    raise RuntimeError('Full-vocabulary suite did not finish')
                time.sleep(2)
            with (out / 'recall-320k.log').open('w') as f:
                subprocess.run([sys.executable, str(HERE / 'recall.py'), '--tokens', '320000',
                                '--fixed-repetitions', '12304', '--port', '19312', '--model', 'velo-yarn-audit',
                                '--out', str(out / 'recall-320k.json')], stdout=f, stderr=subprocess.STDOUT, check=True)
            (out / 'completed.json').write_text(json.dumps({'wall': time.time(), 'qualification': '320K recall; 1M configured KV allocation and saturated checkpoint cap'}))
            print((out / 'suite-summary.json').read_text(), flush=True)
        finally:
            (out / 'stop').touch()
            deadline = time.monotonic() + 40
            while not (out / 'exit.json').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Full-vocabulary trial did not stop')
                time.sleep(1)
        return
    deadline = time.monotonic() + 3600
    while not ((a.root / 'completed.json').exists() and (a.root / 'velo/exit.json').exists()):
        if time.monotonic() > deadline:
            raise TimeoutError('Main A/B did not finish')
        time.sleep(2)
    out = a.root / 'velo-full-vocab'
    with (a.root / 'velo-full-vocab-launch.log').open('w') as log:
        trial = subprocess.Popen([
            sys.executable, str(HERE / 'trial.py'), '--root', str(a.root), '--label', out.name,
            '--context', '1048576', '--factor', '4', '--port', '19312',
            '--prefix-cache', 'on', '--prefix-ckpt-mem-gb', '4', '--cpuset', '5-9,15-19',
            '--mtp-head-n', '0',
        ], stdout=log, stderr=subprocess.STDOUT)
        try:
            while not (out / 'short-complete').exists():
                if trial.poll() is not None:
                    raise RuntimeError(f'Full-vocabulary startup failed: {trial.returncode}')
                time.sleep(2)
            with (out / 'suite.log').open('w') as f:
                subprocess.run([sys.executable, str(HERE / 'compare.py'), '--base', 'http://127.0.0.1:19312',
                                '--model', 'velo-yarn-audit', '--out', str(out)], stdout=f, stderr=subprocess.STDOUT, check=True)
            with (out / 'recall-320k.log').open('w') as f:
                subprocess.run([sys.executable, str(HERE / 'recall.py'), '--tokens', '320000',
                                '--fixed-repetitions', '12304', '--port', '19312', '--model', 'velo-yarn-audit',
                                '--out', str(out / 'recall-320k.json')], stdout=f, stderr=subprocess.STDOUT, check=True)
            (out / 'completed.json').write_text(json.dumps({'wall': time.time(), 'qualification': '320K recall; 1M configured KV allocation and saturated checkpoint cap'}))
            print((out / 'suite-summary.json').read_text(), flush=True)
        finally:
            (out / 'stop').touch()
            trial.wait(timeout=40)


if __name__ == '__main__':
    main()
