#!/usr/bin/env python3
"""Recheck existing TP1 image sequentially; never restarts after host reboot.

Requires all model containers stopped and the existing sglang-qwen38-fn
container as the source of mounts/resources. Saves synthetic outputs durably.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / '2026-09-06-recheck'
IMAGE = 'dgx-sglang-qwen38-fn:sm121-vocab1'
STOP = threading.Event()
PHASE = 'initial'
BOOT = Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def save(name, data):
    temporary = OUT / (name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(OUT / name)


def monitor():
    with (OUT / 'telemetry.jsonl').open('a', buffering=1) as handle:
        while not STOP.is_set():
            fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            record = dict(time=datetime.datetime.now().astimezone().isoformat(), phase=PHASE,
                          boot_id=BOOT, available_gib=int(fields['MemAvailable'].split()[0])/1048576)
            try:
                record['gpu'] = run('nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu', '--format=csv,noheader,nounits')
            except subprocess.CalledProcessError as exc:
                record['gpu_error'] = str(exc)
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
            if record['available_gib'] < 8:
                save('abort.json', record)
                STOP.set()
            STOP.wait(10)


def pause(seconds):
    if STOP.wait(seconds):
        raise RuntimeError('monitor requested stop')


def main():
    global PHASE
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'run.json').exists():
        raise RuntimeError('Existing run: use a new result directory; do not overwrite')
    if run('docker', 'ps', '-q'):
        raise RuntimeError('Stop existing containers before this isolated comparison')
    base = json.loads(run('docker', 'inspect', 'sglang-qwen38-fn'))[0]
    save('run.json', dict(boot_id=BOOT, started=datetime.datetime.now().astimezone().isoformat(), image=run('docker','image','inspect',IMAGE,'--format','{{.Id}}'), rounds=3, seed=595305611, idle_seconds=300))
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for mode in ['off', 'ko64k']:
            name = 'qwen38-tp1-recheck-' + mode
            PHASE = mode + ':loading'
            host = base['HostConfig']
            args = ['docker','run','-d','--name',name,'--restart=no','--init','--gpus','all','--ipc=host','--shm-size=16g','--memory',str(host['Memory']),'--memory-swap',str(host['MemorySwap']),'--cpuset-cpus',host['CpusetCpus'],'--network',host['NetworkMode'],'-p','127.0.0.1:8000:30000']
            for mount in host['Binds']:
                args += ['-v', mount]
            for env in base['Config']['Env']:
                if env.split('=',1)[0] in ['HF_HOME','HF_HUB_OFFLINE','PYTHONUNBUFFERED','PYTORCH_CUDA_ALLOC_CONF','TORCHINDUCTOR_CACHE_DIR','MAX_JOBS','TORCHINDUCTOR_COMPILE_THREADS','SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB']:
                    args += ['-e',env]
            command = base['Config']['Cmd'][:]
            if '--random-seed' in command:
                index=command.index('--random-seed'); del command[index:index+2]
            args += ['-e','SPARKTALK_FLASH_NEXT_DRAFT_VOCAB='+mode,'--entrypoint','python3',IMAGE,'/opt/sparktalk-flash-next/launch.py',*command,'--random-seed','595305611']
            save(mode+'-launch.json', args)
            print(PHASE, run(*args), flush=True)
            try:
                ready = False
                for _ in range(180):
                    pause(10)
                    if run('docker','inspect',name,'--format','{{.State.Running}}') != 'true':
                        raise RuntimeError(name+' exited during load')
                    try:
                        ready=requests.get('http://127.0.0.1:8000/health',timeout=2).ok
                    except requests.RequestException:
                        pass
                    if ready:
                        break
                if not ready:
                    raise RuntimeError('model load timeout')
                info=requests.get('http://127.0.0.1:8000/get_server_info',timeout=10).json()
                save(mode+'-server.json',info)
                PHASE=mode+':benchmark'
                print(PHASE,flush=True)
                with (OUT/(mode+'-benchmark.log')).open('w') as log:
                    process=subprocess.Popen([sys.executable,str(ROOT/'compare.py'),'--rounds','3','--output',str(OUT/(mode+'.json'))],stdout=log,stderr=subprocess.STDOUT)
                    try:
                        for _ in range(180):
                            if process.poll() is not None: break
                            pause(10)
                        if process.poll() is None: raise RuntimeError('benchmark timeout')
                        if process.returncode: raise RuntimeError('benchmark failed: see log')
                    finally:
                        if process.poll() is None:
                            process.terminate(); process.wait(timeout=20)
                PHASE=mode+':idle'
                print(PHASE,flush=True)
                pause(300)
                if not requests.get('http://127.0.0.1:8000/health',timeout=10).ok:
                    raise RuntimeError('idle health failed')
                save(mode+'-idle.json',dict(passed=True,seconds=300,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
            finally:
                PHASE=mode+':stopping'
                run('docker','stop','-t','30',name)
                save(mode+'-state.json',json.loads(run('docker','inspect',name))[0]['State'])
                with (OUT/(mode+'-server.log')).open('w') as log:
                    subprocess.run(['docker','logs','--timestamps',name],stdout=log,stderr=subprocess.STDOUT)
            pause(20)
        PHASE='complete'
        save('complete.json',dict(finished=datetime.datetime.now().astimezone().isoformat(),boot_id=BOOT))
        print('complete',flush=True)
    except Exception as exc:
        save('error.json',dict(phase=PHASE,error=str(exc)))
        raise
    finally:
        STOP.set(); thread.join(timeout=15)


if __name__ == '__main__':
    main()
