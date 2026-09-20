"""Isolated vLLM trial with explicit host-memory guard and persisted evidence."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
NAME = 'qwen38-qad-vllm-trial'

def command(*args):
    return subprocess.check_output(args, text=True).strip()

def memory():
    values = dict((line.split(':')[0], int(line.split()[1]))
                  for line in Path('/proc/meminfo').read_text().splitlines())
    return {'available_gib': values['MemAvailable']/2**20,
            'swap_used_gib': (values['SwapTotal']-values['SwapFree'])/2**20}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--compose', type=Path, default=HERE/'compose.context1m.yaml')
    parser.add_argument('--probe', type=Path, default=HERE/'probe.py')
    parser.add_argument('--probe-arg', action='append', default=[])
    parser.add_argument('--min-available-gib', type=float, default=3)
    args = parser.parse_args()
    if command('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'):
        raise RuntimeError('Existing GPU compute process; no trial started')
    if memory()['available_gib'] < 95:
        raise RuntimeError('Insufficient starting memory')
    if NAME in command('docker', 'ps', '--format', '{{.Names}}').splitlines():
        raise RuntimeError('Trial already running; will not replace it')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    compose = ['docker','compose','-f',str(args.compose.resolve())]
    start = time.monotonic()
    since = datetime.datetime.now(datetime.timezone.utc).isoformat()
    probe = None
    outcome = {'status':'starting','minimum_available_gib':memory()['available_gib'],
               'guard_gib':args.min_available_gib}
    try:
        subprocess.run(compose+['up','-d','--no-build'],check=True)
        inspect = json.loads(command('docker','inspect',NAME))[0]
        env = dict(s.split('=',1) for s in inspect['Config']['Env'])
        keys = ['VLLM_PLE_TABLE_MEMORY','CUTE_DSL_ARCH','SAFETENSORS_FAST_GPU',
                'VLLM_WORKER_MULTIPROC_METHOD','VLLM_SSM_CONV_STATE_LAYOUT',
                'VLLM_USE_AOT_COMPILE','VLLM_USE_MEGA_AOT_ARTIFACT',
                'VLLM_USE_V2_MODEL_RUNNER','B12X_POLICY_MODE','TMPDIR']
        manifest = {'image':inspect['Image'],'command':inspect['Config']['Cmd'],
                    'environment':{k:env.get(k) for k in keys},'guard_gib':args.min_available_gib}
        (args.output_dir/'runtime.json').write_text(json.dumps(manifest,indent=2))
        with (args.output_dir/'memory.jsonl').open('w') as samples, (args.output_dir/'api-probes.log').open('w') as output:
            while True:
                elapsed = time.monotonic()-start
                sample = dict(seconds=round(elapsed,2),**memory())
                samples.write(json.dumps(sample)+'\n'); samples.flush()
                outcome['minimum_available_gib'] = min(outcome['minimum_available_gib'],sample['available_gib'])
                if sample['available_gib'] < args.min_available_gib:
                    outcome['status']='memory_guard_stop'
                    raise RuntimeError(f'Available memory {sample["available_gib"]:.3f} GiB below {args.min_available_gib} GiB')
                state = json.loads(command('docker','inspect','--format','{{json .State}}',NAME))
                if not state['Running']:
                    outcome['status']='server_exit'
                    outcome['server_state_before_stop']=state
                    raise RuntimeError(f'Server exited: {state}')
                if probe is None:
                    ready=False
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8017/health',timeout=2) as r:
                            ready=r.status==200
                    except OSError:
                        pass
                    if ready:
                        outcome['healthy_seconds']=elapsed
                        outcome['status']='probing'
                        print(f'Healthy at {elapsed:.1f}s; starting probes',flush=True)
                        probe=subprocess.Popen([sys.executable,str(args.probe.resolve()),'--output',str(args.output_dir/'api-results.json'),*args.probe_arg],stdout=output,stderr=subprocess.STDOUT)
                    elif elapsed>3600:
                        outcome['status']='startup_timeout'
                        raise TimeoutError('Startup exceeded 3600s')
                elif probe.poll() is not None:
                    if probe.returncode:
                        outcome['status']='probe_failed'
                        raise RuntimeError('API probe failed; see api-probes.log')
                    outcome['status']='passed'
                    print('All probes passed',flush=True)
                    return
                time.sleep(2)
    except Exception as e:
        outcome['error']=str(e)
        raise
    finally:
        if probe is not None and probe.poll() is None:
            probe.terminate();probe.wait(timeout=10)
        with (args.output_dir/'server-at-stop.log').open('w') as output:
            subprocess.run(['docker','logs','--since',since,NAME],stdout=output,stderr=subprocess.STDOUT)
        subprocess.run(compose+['stop','-t','30'],check=False)
        with (args.output_dir/'server.log').open('w') as output:
            subprocess.run(['docker','logs','--since',since,NAME],stdout=output,stderr=subprocess.STDOUT)
        outcome['elapsed_seconds']=time.monotonic()-start
        outcome['final_state']=json.loads(command('docker','inspect','--format','{{json .State}}',NAME))
        (args.output_dir/'outcome.json').write_text(json.dumps(outcome,indent=2))

if __name__ == '__main__':
    main()
