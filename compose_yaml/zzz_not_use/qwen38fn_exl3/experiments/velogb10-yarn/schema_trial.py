"""Run schema qualification with the cached EXL3 weights; stop every owned server."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True); a = ap.parse_args()
    for label, context, width, port in [('single', 1048576, 1, 19320), ('batched', 65536, 2, 19321)]:
        out = a.root / label
        with (a.root / (label + '-launcher.log')).open('w') as log:
            proc = subprocess.Popen([sys.executable, str(HERE / 'trial.py'), '--root', str(a.root), '--label', label,
                '--context', str(context), '--factor', '4', '--port', str(port), '--max-batch', str(width),
                '--prefix-cache', 'on', '--prefix-ckpt-mem-gb', '4', '--cpuset', '5-9,15-19', '--mtp-head-n', '0',
                '--pdl', '0' if label == 'single' else '1'],
                stdout=log, stderr=subprocess.STDOUT)
            try:
                while not (out / 'short-complete').exists():
                    if proc.poll() is not None: raise RuntimeError(f'{label} server failed during startup')
                    time.sleep(2)
                print('READY', label, flush=True)
                cmd = [sys.executable, str(HERE / 'schema_probe.py'), '--base', f'http://127.0.0.1:{port}', '--out', str(out)]
                if label == 'batched': cmd.append('--concurrent-only')
                with (out / 'schema-probe.log').open('w') as f:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
                (out / 'memory-after-schema.json').write_text((out / 'memory.jsonl').read_text().splitlines()[-1])
                if label == 'single':
                    with (out / 'suite.log').open('w') as f:
                        subprocess.run([sys.executable, str(HERE / 'compare.py'), '--base', f'http://127.0.0.1:{port}',
                                        '--model', 'velo-yarn-audit', '--out', str(out)], stdout=f, stderr=subprocess.STDOUT, check=True)
                    (out / 'memory-after-suite.json').write_text((out / 'memory.jsonl').read_text().splitlines()[-1])
                    with (out / 'recall-320k.log').open('w') as f:
                        subprocess.run([sys.executable, str(HERE / 'recall.py'), '--tokens', '320000',
                                        '--fixed-repetitions', '12304', '--port', str(port), '--model', 'velo-yarn-audit',
                                        '--out', str(out / 'recall-320k.json')], stdout=f, stderr=subprocess.STDOUT, check=True)
                    (out / 'memory-after-320k.json').write_text((out / 'memory.jsonl').read_text().splitlines()[-1])
                print(label, (out / 'schema-summary.json').read_text(), flush=True)
            finally:
                (out / 'stop').touch()
                proc.wait(timeout=40)
    (a.root / 'completed.json').write_text(json.dumps({'wall': time.time()}))


if __name__ == '__main__': main()
