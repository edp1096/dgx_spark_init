"""Qualify a completed worker candidate using the existing comparison probes."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def run(args):
    root = Path(__file__).resolve().parents[4]
    evaluate = module('evaluate', Path(__file__).parent.parent/'qwen38_huihui/evaluate.py')
    probe = module('probe_context', root/'compose_yaml/qwen38_fn_sglang/experiments/lil-qad-tp1/probe_context.py')
    args.output.mkdir(parents=True, exist_ok=True)
    def remote(command):
        return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', args.worker, shlex.join(command)], text=True)
    base = f'http://{args.worker.split("@")[-1]}:{args.port}'
    model = args.job+'/outputs/Huihui-Qwen3.8-Flash-Next-NVFP4-QAD5500-BF16Delta'
    # A final directory is created only after byte preservation and hash checks.
    deadline = time.monotonic()+4*3600
    while True:
        state = remote(['python3', '-c',
            'import pathlib,json; p=pathlib.Path('+repr(model+'/transfer-manifest.json')+'); '
            'print(json.loads(p.read_text())["status"] if p.exists() else "pending")']).strip()
        if state == 'candidate_verified':
            break
        if state != 'pending' or time.monotonic()>deadline:
            raise RuntimeError(('Candidate verification did not complete', state))
        time.sleep(15)
    summary = {'status': 'running', 'scopes': {}}
    for context in [65536, 1048576]:
        name = 'qad5500-ablit-'+str(context)
        out = args.output/str(context)
        out.mkdir(exist_ok=True)
        print('LAUNCH', context, flush=True)
        remote(['python3', args.job+'/tools/adapter/launch_validation.py',
            '--model', model, '--cache', args.job+'/runtime-baseline',
            '--name', name, '--port', str(args.port), '--context', str(context)])
        try:
            for _ in range(180):
                try:
                    with urllib.request.urlopen(base+'/v1/models', timeout=3) as response:
                        if response.status==200:
                            break
                except Exception:
                    pass
                time.sleep(5)
            else:
                raise RuntimeError('Candidate startup timed out')
            print('READY', context, flush=True)
            evaluate.main(base, out/'smoke.json', name)
            smoke=json.loads((out/'smoke.json').read_text())
            if any(r.get('passed') is False or r.get('error') for r in smoke['results']):
                raise RuntimeError('Candidate smoke assertions failed; inspect responses')
            probe.BASE, probe.MODEL = base, name
            old_args=sys.argv
            try:
                sys.argv=['probe_context.py', '--output', str(out/'retrieval.json'),
                          '--targets', str(context), '--diverse']
                probe.main()
            finally:
                sys.argv=old_args
            # Exercise short output after the large cache allocation, too.
            response=probe.request('/v1/chat/completions', {
                'model':name, 'messages':[{'role':'user','content':'17과 25를 더한 값을 숫자로만 답하세요.'}],
                'temperature':0, 'max_tokens':32, 'chat_template_kwargs':{'enable_thinking':False}})
            (out/'post-long-smoke.json').write_text(json.dumps(response,ensure_ascii=False,indent=2))
            if response['choices'][0]['message']['content'].strip()!='42':
                raise RuntimeError('Short generation after long input failed')
            summary['scopes'][str(context)]='passed'
        finally:
            (out/'server.log').write_text(remote(['docker','logs',name]))
            remote(['docker','stop','-t','30',name])
            # Retain stopped validation containers for inspection.
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    summary['status']='passed'
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('RUNTIME VALIDATION PASSED', flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--worker',required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=30125)
    run(p.parse_args())
