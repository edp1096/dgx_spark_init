"""Boot, probe, and stop only the isolated trial; never replace Talk services."""
import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
NAME = 'qwen38-qad-sglang-trial'
SNAPSHOT = pathlib.Path.home() / (
    '.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/'
    'snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd')


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def available_gib():
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    raise RuntimeError('MemAvailable unavailable')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--compose', type=pathlib.Path, default=HERE/'compose.yaml')
    parser.add_argument('--probe', type=pathlib.Path, default=HERE/'smoke_test.py')
    parser.add_argument('--probe-arg', action='append', default=[])
    parser.add_argument('--min-available-gib', type=float, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(HERE/'check_checkpoint.py'), str(SNAPSHOT)], check=True)
    active = command('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader').strip()
    if active:
        raise RuntimeError('GPU compute processes are active; trial was not started')
    if available_gib() < 95:
        raise RuntimeError('Trial needs at least 95 GiB available host memory at startup')
    running = command('docker', 'ps', '--format', '{{.Names}}').splitlines()
    if NAME in running:
        raise RuntimeError('Trial already running; refusing to replace it')

    probe = None
    started = time.monotonic()
    log_since = datetime.datetime.now(datetime.timezone.utc).isoformat()
    minimum = available_gib()
    samples = []
    compose = ['docker', 'compose', '-f', str(args.compose.resolve())]
    try:
        subprocess.run(compose + ['up', '-d', '--no-build'], check=True)
        # Record the immutable image ID and effective arguments, not a moving tag.
        manifest = json.loads(command('docker', 'inspect', '--format',
                           '{"image":{{json .Image}},"command":{{json .Config.Cmd}}}', NAME))
        keys = {'PYTORCH_CUDA_ALLOC_CONF', 'SPARKTALK_FLASH_NEXT_DRAFT_VOCAB',
                'SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN',
                'SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB',
                'SGLANG_QAD_B12X_GDN', 'SGLANG_QAD_PLE_IO_URING', 'B12X_POLICY_MODE'}
        env = json.loads(command('docker', 'inspect', '--format', '{{json .Config.Env}}', NAME))
        manifest['environment'] = dict(item.split('=',1) for item in env if item.split('=',1)[0] in keys)
        manifest['minimum_host_available_gib'] = args.min_available_gib
        (args.output_dir/'runtime.json').write_text(json.dumps(manifest, indent=2)+'\n')
        ready = False
        with (args.output_dir/'api-probes.log').open('w') as output:
            while True:
                elapsed = time.monotonic()-started
                available = available_gib()
                minimum = min(minimum, available)
                samples.append({'seconds': round(elapsed, 1), 'available_gib': round(available, 2)})
                if available < args.min_available_gib:
                    raise RuntimeError(f'Host available memory below {args.min_available_gib:g} GiB; stopping only the trial')
                state = json.loads(command('docker', 'inspect', '--format', '{{json .State}}', NAME))
                if not state['Running']:
                    raise RuntimeError(f'Trial exited: code={state["ExitCode"]}, OOM={state["OOMKilled"]}')
                if not ready:
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8016/health', timeout=2) as response:
                            ready = response.status == 200
                    except (OSError, urllib.error.URLError):
                        pass
                    if ready:
                        print(f'Trial healthy after {elapsed:.1f}s; starting API probes', flush=True)
                        probe = subprocess.Popen([
                            sys.executable, str(args.probe.resolve()), '--output',
                            str(args.output_dir/'api-results.json'), *args.probe_arg], stdout=output, stderr=subprocess.STDOUT)
                    elif elapsed > 1800:
                        raise TimeoutError('Trial startup exceeded 1800 seconds')
                if probe is not None and probe.poll() is not None:
                    if probe.returncode:
                        raise RuntimeError(f'API probes failed: see {args.output_dir}/api-probes.log')
                    print(f'API probes passed; minimum host available memory {minimum:.2f} GiB', flush=True)
                    return
                time.sleep(5)
    finally:
        if probe is not None and probe.poll() is None:
            probe.terminate()
            probe.wait(timeout=10)
        with (args.output_dir/'server.log').open('w') as output:
            subprocess.run(['docker', 'logs', '--since', log_since, NAME], stdout=output, stderr=subprocess.STDOUT)
        subprocess.run(compose + ['stop', '-t', '30'], check=False)
        (args.output_dir/'memory.json').write_text(json.dumps(samples, indent=2))


if __name__ == '__main__':
    main()
