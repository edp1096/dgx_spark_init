"""Finish loader-only control, then FP8 candidate, and stop only trial services."""
import json
import subprocess
import sys
import time
from pathlib import Path
import trial


def wait_ready(label):
    started=time.monotonic()
    while time.monotonic()-started < 1800:
        try:
            info=trial.request('/get_server_info',timeout=3)
            if info.get('status')=='ready':
                assert info['internal_states'][0]['qad_memory']['capacity_tokens']==1048576
                with trial.OPENER.open(trial.BASE+'/health',timeout=30) as response:
                    assert response.status==200
                print('SCHEDULER_READY',label,flush=True)
                return
        except (OSError,ValueError,TimeoutError):pass
        state=json.loads(trial.command('docker','inspect',trial.NAME))[0]['State']
        if not state['Running']:raise RuntimeError(f'Trial exited: {label}: {state}')
        time.sleep(2)
    raise TimeoutError(label)


if __name__=='__main__':
    for label in ['baseline-opt','hybrid-opt']:
        if label=='hybrid-opt':
            trial.start(label)
            log=(trial.OUT/label/'monitor.log').open('w')
            subprocess.Popen([sys.executable,str(trial.HERE/'trial.py'),'monitor',label],stdout=log,stderr=subprocess.STDOUT)
        wait_ready(label)
        if label.startswith('hybrid'):
            logs=trial.command('docker','logs',trial.NAME)
            assert logs.count('RADIXARK_FP8_VERIFIED ')==192, 'FP8 conversion was not fully loaded'
        trial.bench(label)
        subprocess.run([sys.executable,str(trial.HERE/'quality.py'),label],check=True)
        print('BENCH_COMPLETE',label,flush=True)
        subprocess.run([sys.executable,str(trial.HERE/'trial.py'),'stop',label],check=True)
    print('SUITE_COMPLETE',flush=True)
