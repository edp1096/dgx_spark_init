#!/usr/bin/env python3
"""Sequential, guarded draft-vocabulary memory trials on one DGX Spark.

Requires a stopped production container, pre-generated maps and a measurement-only
worker overlay in OUTPUT/.runtime. Never starts/stops the production container or
changes SparkTalk settings. Estimate.json must exist before any model is started.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

import requests

ROOT = Path(__file__).resolve().parent
GIB = 1024 ** 3
SPEC_WORKER = '/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py'
ENV_KEYS = {'HF_HOME', 'HF_HUB_OFFLINE', 'PYTHONUNBUFFERED',
            'PYTORCH_CUDA_ALLOC_CONF', 'TORCHINDUCTOR_CACHE_DIR', 'MAX_JOBS',
            'TORCHINDUCTOR_COMPILE_THREADS', 'SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB'}
STATUS_HELPER_IMAGE = 'sha256:3e9f3890915f15b524dcab671357bb6c3d87014c2bed09d0388880a642711428'


def command(*args, timeout=20):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT,
                                   timeout=timeout).strip()


def memory():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {k: int(values[k].split()[0]) * 1024 for k in
            ['MemTotal', 'MemAvailable', 'MemFree', 'SwapTotal', 'SwapFree', 'Cached']}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def is_status_helper(container):
    """Only the verified, network-isolated CPU key-store status helper is exempt."""
    host, config = container['HostConfig'], container['Config']
    return (container['Image'] == STATUS_HELPER_IMAGE
            and config.get('Entrypoint') == ['/usr/local/bin/sparktalk-extra-ssh']
            and config.get('Cmd') == ['-key-store', 'status']
            and host.get('NetworkMode') == 'none'
            and host.get('Runtime') in ('runc', '')
            and not host.get('Privileged')
            and not host.get('DeviceRequests') and not host.get('Devices')
            and all(not bind.split(':', 1)[0].startswith('/dev') for bind in host.get('Binds', [])))


def running():
    result = set()
    for line in command('docker', 'ps', '--format', '{{.ID}} {{.Image}}').splitlines():
        ident, image = line.split(' ', 1)
        if image == 'sparktalk-extra-ssh:latest':
            inspection = subprocess.run(['docker', 'inspect', ident], text=True,
                                        capture_output=True, timeout=10)
            if inspection.returncode:
                # A --rm helper can disappear between ps and inspect.
                if ident not in command('docker', 'ps', '--format', '{{.ID}}').splitlines():
                    continue
                raise RuntimeError('could not inspect a newly running container')
            if is_status_helper(json.loads(inspection.stdout)[0]):
                continue
        result.add(ident)
    return result


def set_option(args, key, value):
    if key in args:
        i = args.index(key); del args[i:i + 2]
    args += [key, str(value)]


class Trial:
    def __init__(self, out, mode, baseline, allowed, floor):
        self.out, self.mode, self.baseline = out, mode, baseline
        self.allowed, self.floor = allowed, floor * GIB
        self.name = f'qwen-fn-vocab-memory-{mode}'
        self.phase = 'loading'
        self.done = threading.Event()
        self.aborted = threading.Event()
        self.abort_reason = ''
        self.records = []
        self.cg = None
        self.cid = None
        self.created = False
        self.log_process = None
        self.guard = None
        self.monitor_error = None
        self.boot = Path('/proc/sys/kernel/random/boot_id').read_text()

    def abort(self, reason):
        if self.aborted.is_set():
            return
        self.abort_reason = reason
        self.aborted.set()
        print(f'ABORT {self.mode}: {reason}', flush=True)
        # Stop first; failed telemetry/disk writes must never prevent shutdown.
        try:
            if self.created:
                subprocess.run(['docker', 'kill', self.name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10)
        finally:
            try:
                save(self.out / f'{self.mode}-abort.json', dict(reason=reason, memory=memory()))
            except Exception as exc:
                print(f'Abort telemetry unavailable: {exc}', flush=True)

    def monitor(self):
        try:
            with (self.out / f'{self.mode}-telemetry.jsonl').open('a', buffering=1) as stream:
                while not self.done.is_set():
                    record = dict(time=datetime.datetime.now().astimezone().isoformat(),
                                  monotonic=time.monotonic(), phase=self.phase, **memory())
                    if self.cg:
                        for key in ['memory.current', 'memory.peak', 'memory.events']:
                            try:
                                raw = (self.cg / key).read_text().strip()
                                record[key] = (dict(line.split() for line in raw.splitlines())
                                               if key == 'memory.events' else int(raw))
                            except (FileNotFoundError, PermissionError):
                                pass
                    self.records.append(record)
                    stream.write(json.dumps(record) + '\n')
                    if record['MemAvailable'] < self.floor:
                        self.abort(f'MemAvailable {record["MemAvailable"] / GIB:.2f} GiB below {self.floor / GIB:.0f} GiB')
                    if record['SwapFree'] < self.baseline['SwapFree'] - 512 * 1024 ** 2:
                        self.abort('Swap usage grew by more than 512 MiB')
                    if Path('/proc/sys/kernel/random/boot_id').read_text() != self.boot:
                        self.abort('host boot changed')
                    self.done.wait(0.25)
                stream.flush(); os.fsync(stream.fileno())
        except Exception as exc:
            self.monitor_error = repr(exc)
            self.abort('memory monitor failed: ' + repr(exc))

    def check(self):
        if self.aborted.is_set():
            raise RuntimeError(self.abort_reason)
        unexpected = running() - self.allowed - {self.cid[:12]}
        if unexpected:
            self.abort('another container was started during the isolated trial')
            raise RuntimeError(self.abort_reason)
        state = json.loads(command('docker', 'inspect', self.name))[0]['State']
        if not state['Running']:
            raise RuntimeError(f'{self.mode} exited: {state}')

    def report(self):
        path = self.out / f'{self.mode}-server.log'
        text = ''
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 20000))
                text = stream.read().decode(errors='replace')
        shards = re.findall(r'(\d+/206)', text)
        print(f'{self.mode}: {self.phase}; available={memory()["MemAvailable"] / GIB:.2f} GiB; '
              f'shards={shards[-1] if shards else "pending"}', flush=True)

    def wait_process(self, process, timeout):
        deadline = time.monotonic() + timeout
        next_report = 0
        while process.poll() is None:
            self.check()
            if time.monotonic() > deadline:
                raise RuntimeError('benchmark timeout')
            if time.monotonic() >= next_report:
                self.report(); next_report = time.monotonic() + 30
            time.sleep(2)
        if process.returncode:
            raise RuntimeError(f'benchmark failed with code {process.returncode}')

    def run(self, source, port, kv_tokens, mamba_slots, workload=None, workload_timeout=1200, rounds=1):
        host = source['HostConfig']
        args = ['docker', 'run', '-d', '--name', self.name, '--restart=no', '--init',
                '--gpus', 'all', '--ipc=host', '--shm-size=16g',
                '--memory', str(host['Memory']), '--memory-swap', str(host['MemorySwap']),
                '--cpuset-cpus', host['CpusetCpus'], '-p', f'127.0.0.1:{port}:30000']
        for mount in host['Binds']:
            args += ['-v', mount]
        args += ['-v', f'{self.out / ".runtime"}:/vocab-memory:ro',
                 '-v', f'{self.out / ".runtime/eagle_worker_v2.py"}:{SPEC_WORKER}:ro']
        for env in source['Config']['Env']:
            if env.split('=', 1)[0] in ENV_KEYS:
                args += ['-e', env]
        cmd = source['Config']['Cmd'][:]
        for key, value in [('--context-length', 65536), ('--max-total-tokens', kv_tokens),
                           ('--max-mamba-cache-size', mamba_slots), ('--random-seed', 595305611)]:
            set_option(cmd, key, value)
        if '--speculative-token-map' in cmd:
            i = cmd.index('--speculative-token-map'); del cmd[i:i+2]
        if self.mode != 'full':
            cmd += ['--speculative-token-map', f'/vocab-memory/{self.mode}.pt']
        args += ['-e', 'SPARKTALK_FLASH_NEXT_DRAFT_VOCAB=off', '--entrypoint', 'python3',
                 source['Config']['Image'], '/opt/sparktalk-flash-next/launch.py', *cmd]
        save(self.out / f'{self.mode}-launch.json', args)
        bench = None
        with (self.out / f'{self.mode}-server.log').open('w') as log:
            self.guard = threading.Thread(target=self.monitor, daemon=True)
            self.guard.start()
            try:
                self.cid = command(*args)
                self.created = True
                state = json.loads(command('docker', 'inspect', self.name))[0]['State']
                cg = Path(f'/proc/{state["Pid"]}/cgroup').read_text().strip().split('::', 1)[1]
                self.cg = Path('/sys/fs/cgroup') / cg.lstrip('/')
                self.log_process = subprocess.Popen(['docker', 'logs', '-f', '--timestamps', self.name],
                                                    stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 1800
                next_report = 0
                while True:
                    self.check()
                    if time.monotonic() > deadline:
                        raise RuntimeError('load timeout')
                    try:
                        if requests.get(f'http://127.0.0.1:{port}/health', timeout=2).ok:
                            break
                    except requests.RequestException:
                        pass
                    if time.monotonic() >= next_report:
                        self.report(); next_report = time.monotonic() + 30
                    time.sleep(3)
                save(self.out / f'{self.mode}-server-info.json',
                     requests.get(f'http://127.0.0.1:{port}/get_server_info', timeout=10).json())
                self.phase = 'ready_idle'
                for _ in range(15):
                    self.check(); time.sleep(2)
                self.phase = 'benchmark'
                with (self.out / f'{self.mode}-benchmark.log').open('w') as benchmark_log:
                    bench = subprocess.Popen([sys.executable, str(workload or ROOT / 'compare.py'), '--url',
                                              f'http://127.0.0.1:{port}', '--rounds', str(rounds), '--output',
                                              str(self.out / f'{self.mode}-benchmark.json')],
                                             stdout=benchmark_log, stderr=subprocess.STDOUT)
                    self.wait_process(bench, workload_timeout)
                self.phase = 'post_idle'
                for _ in range(15):
                    self.check(); time.sleep(2)
                assert requests.get(f'http://127.0.0.1:{port}/health', timeout=5).ok
                print(f'{self.mode}: all memory workload phases completed', flush=True)
            finally:
                if bench and bench.poll() is None:
                    bench.terminate()
                    try:
                        bench.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        bench.kill(); bench.wait()
                self.phase = 'stopping'
                if self.created:
                    subprocess.run(['docker', 'stop', '-t', '10', self.name], timeout=20,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    state = json.loads(command('docker', 'inspect', self.name))[0]['State']
                    save(self.out / f'{self.mode}-container-state.json', state)
                    if state.get('OOMKilled'):
                        self.abort('container reports OOMKilled')
                if self.log_process:
                    self.log_process.wait(timeout=15)
                self.done.set()
                if self.guard:
                    self.guard.join(timeout=15)
                self.summarize()
                if self.created:
                    command('docker', 'rm', self.name)
        if self.aborted.is_set():
            raise RuntimeError(self.abort_reason)

    def summarize(self):
        grouped = {}
        for phase in ['loading', 'ready_idle', 'benchmark', 'post_idle']:
            rows = [r for r in self.records if r['phase'] == phase]
            if not rows:
                continue
            available = [r['MemAvailable'] for r in rows]
            grouped[phase] = dict(samples=len(rows), min_available_gib=min(available)/GIB,
                                  median_available_gib=statistics.median(available)/GIB,
                                  max_extra_memory_gib=(self.baseline['MemAvailable']-min(available))/GIB,
                                  median_extra_memory_gib=(self.baseline['MemAvailable']-statistics.median(available))/GIB)
        text = (self.out / f'{self.mode}-server.log').read_text(errors='replace')
        heads = [json.loads(m) for m in re.findall(r'VOCAB_MEMORY_HEAD (\{[^\n]+\})', text)]
        cache = [line for line in text.splitlines() if any(key in line for key in
                 ['KV Cache is allocated', 'Mamba Cache is allocated', 'max_total_num_tokens='])]
        save(self.out / f'{self.mode}-memory.json', dict(mode=self.mode, baseline=self.baseline,
             phases=grouped, head_measurements=heads, cache_logs=cache,
             aborted=self.aborted.is_set(), abort_reason=self.abort_reason, monitor_error=self.monitor_error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', default='sglang-qwen38-fn')
    parser.add_argument('--port', type=int, default=18000)
    parser.add_argument('--kv-tokens', type=int, default=65536)
    parser.add_argument('--mamba-slots', type=int, default=18)
    parser.add_argument('--min-available-gib', type=float, default=16)
    parser.add_argument('--modes', nargs='+', choices=['ko16k', 'ko32k', 'ko64k', 'ko128k', 'full'],
                        default=['ko64k', 'ko128k', 'full'])
    parser.add_argument('--workload', type=Path)
    parser.add_argument('--workload-timeout', type=int, default=1200)
    parser.add_argument('--rounds', type=int, default=1)
    args = parser.parse_args()
    out = args.output.resolve()
    if not (out / 'estimate.json').exists() or (out / 'run.json').exists():
        raise RuntimeError('estimate.json required; existing run.json may not be overwritten')
    for name in [f'{mode}.pt' for mode in args.modes if mode != 'full'] + ['eagle_worker_v2.py']:
        if not (out / '.runtime' / name).exists():
            raise RuntimeError('missing prepared input: ' + name)
    source = json.loads(command('docker', 'inspect', args.source))[0]
    if source['State']['Running']:
        raise RuntimeError('production model must be stopped before this isolated test')
    allowed = running()
    for container in json.loads(command('docker', 'inspect', *allowed)) if allowed else []:
        if not container['Name'].startswith('/sparktalk-extra-'):
            raise RuntimeError('only support services may be running')
    save(out / 'run.json', dict(started=datetime.datetime.now().astimezone().isoformat(),
         image=command('docker','image','inspect',source['Config']['Image'],'--format','{{.Id}}'),
         source_container=source['Id'], initial_running_ids=sorted(allowed), context_length=65536,
         kv_tokens=args.kv_tokens, mamba_slots=args.mamba_slots, min_available_gib=args.min_available_gib,
         rounds=args.rounds, modes=args.modes, workload=str(args.workload) if args.workload else 'compare.py',
         workload_timeout=args.workload_timeout))
    current = None
    def interrupted(signum, _frame):
        if current:
            current.abort('interrupted by signal ' + str(signum))
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for mode in args.modes:
            if running() != allowed:
                raise RuntimeError('container state changed between trials')
            baseline = memory()
            if baseline['MemAvailable'] < 110 * GIB:
                raise RuntimeError('at least 110 GiB available is required before each trial')
            current = Trial(out, mode, baseline, allowed, args.min_available_gib)
            current.run(source, args.port, args.kv_tokens, args.mamba_slots,
                        args.workload.resolve() if args.workload else None, args.workload_timeout, args.rounds)
            current = None
            for _ in range(30):
                if memory()['MemAvailable'] >= 110 * GIB:
                    break
                time.sleep(2)
        final = json.loads(command('docker','inspect',args.source))[0]
        assert final['Id'] == source['Id'] and not final['State']['Running']
        assert running() == allowed
        save(out / 'complete.json', dict(finished=datetime.datetime.now().astimezone().isoformat(),
                                        production_container_unchanged=True, support_containers_unchanged=True))
        print('All trials complete; production container remains stopped and support services unchanged', flush=True)
    except BaseException as exc:
        save(out / 'error.json', dict(error=repr(exc), mode=current.mode if current else None))
        raise


if __name__ == '__main__':
    main()
