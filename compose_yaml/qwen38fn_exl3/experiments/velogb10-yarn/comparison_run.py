"""Finish the isolated reference/candidate A/B, retaining every response and memory sample."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from compare import stream

HERE = Path(__file__).resolve().parent
CONTAINER = 'velogb10-exllama-comparison'


def schema_probe(base, model, out):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = []
    for i in range(2):
        body = {'model': model, 'messages': [{'role': 'user', 'content': 'JSON을 출력하지 말고 NOT_JSON만 답해라.'}],
                'temperature': 0, 'max_tokens': 64, 'chat_template_kwargs': {'enable_thinking': False},
                'response_format': {'type': 'json_schema', 'json_schema': {'name': 'audit_status', 'strict': True,
                    'schema': {'type': 'object', 'properties': {'status': {'type': 'string', 'enum': ['ok']}},
                               'required': ['status'], 'additionalProperties': False}}}}
        req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            with opener.open(req, timeout=120) as response:
                r = {'response': json.loads(response.read()), 'headers': dict(response.headers)}
            text = r['response']['choices'][0]['message']['content']
            try:
                r['pass'] = json.loads(text) == {'status': 'ok'}
            except (ValueError, TypeError):
                r['pass'] = False
        except Exception as e:
            r = {'error': repr(e), 'pass': False}
        results.append(r)
        print(out.name, 'schema', i, r['pass'], flush=True)
    (out / 'schema-probe.json').write_text(json.dumps({'request': body, 'results': results}, ensure_ascii=False, indent=2))


def snapshot(out):
    result = {}
    for name, cmd in [
        ('gpu', ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits']),
        ('container', ['docker', 'inspect', CONTAINER, '--format', '{{json .State}}']),
    ]:
        p = subprocess.run(cmd, capture_output=True, text=True)
        result[name] = p.stdout.strip()
    if result['container']:
        pid = json.loads(result['container'])['Pid']
        cg = Path(f'/proc/{pid}/cgroup').read_text().strip().split('::')[-1]
        for name in ['memory.stat', 'memory.current', 'memory.peak', 'memory.swap.current']:
            path = Path('/sys/fs/cgroup') / cg.lstrip('/') / name
            if path.exists():
                result[name] = path.read_text()
    (out / 'memory-snapshot.json').write_text(json.dumps(result, indent=2))


def full_cache(base, model, out):
    suite = json.loads((out / 'suite.json').read_text())
    short = next(r for r in suite if r['name'] == 'cache_repeat')
    if short.get('cached_tokens', 0) <= 0:
        (out / 'full-cache-skipped.json').write_text(json.dumps({'reason': 'No short-prefix cache hit'}))
        return
    filler = 'This entry records a routine maintenance check. All tests passed.\n'
    prompt = ('자료에서 시작, 중간, 끝의 암호를 찾아 마지막 질문에 답하라.\n시작 암호: 청록펭귄.\n'
              + filler * 40319 + '\n중간 암호: 은빛해달.\n' + filler * 40319
              + '\n끝 암호: 자주빛고래.\n질문: 시작, 중간, 끝 암호를 순서대로 모두 출력해라. 암호 세 개만 답해라.')
    messages = [{'role': 'user', 'content': prompt}]
    r = stream(base, model, messages, 96)
    r['pass'] = all(k in r['text'] for k in ['청록펭귄', '은빛해달', '자주빛고래'])
    (out / 'full-cache-repeat.json').write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print(out.name, 'full-cache-repeat', r['ttft'], r['cached_tokens'], r['pass'], flush=True)
    messages += [{'role': 'assistant', 'content': r['text']},
                 {'role': 'user', 'content': '중간 암호만 답해라.'}]
    r = stream(base, model, messages, 32)
    r['pass'] = r['text'].strip().rstrip('.') == '은빛해달'
    (out / 'full-cache-followup.json').write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print(out.name, 'full-cache-followup', r['ttft'], r['cached_tokens'], r['pass'], flush=True)
    r = stream(base, model, [{'role': 'user', 'content': '17×23의 결과를 숫자만 답해라.'}], 32)
    r['pass'] = r['text'].strip() == '391'
    (out / 'post-full-reset.json').write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print(out.name, 'post-full-reset', r['pass'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args()
    ref = a.root / 'exllama'
    velo = a.root / 'velo'
    trial = None
    log = None
    try:
        deadline = time.monotonic() + 3600
        while not (ref / 'recall-1m.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Reference full-context request did not finish')
            time.sleep(2)
        r = json.loads((ref / 'recall-1m.json').read_text())
        print('Reference 1M:', r['seconds'], r['pass'], flush=True)
        full_cache('http://127.0.0.1:19310', 'qwen38fn_exl3', ref)
        schema_probe('http://127.0.0.1:19310', 'qwen38fn_exl3', ref)
        snapshot(ref)
        subprocess.run(['docker', 'stop', '--time', '20', CONTAINER], check=True)
        subprocess.run(['docker', 'rm', CONTAINER], check=True)
        log = (a.root / 'velo-launch.log').open('w')
        trial = subprocess.Popen([
            sys.executable, str(HERE / 'trial.py'), '--root', str(a.root), '--label', 'velo',
            '--context', '1048576', '--factor', '4', '--port', '19311',
            '--prefix-cache', 'on', '--prefix-ckpt-mem-gb', '4', '--cpuset', '5-9,15-19',
        ], stdout=log, stderr=subprocess.STDOUT)
        while not (velo / 'short-complete').exists():
            if trial.poll() is not None:
                raise RuntimeError(f'Candidate startup failed: {trial.returncode}')
            time.sleep(2)
        with (velo / 'suite.log').open('w') as f:
            subprocess.run([sys.executable, str(HERE / 'compare.py'), '--base', 'http://127.0.0.1:19311',
                            '--model', 'velo-yarn-audit', '--out', str(velo)], stdout=f, stderr=subprocess.STDOUT, check=True)
        print('Candidate short suite complete', flush=True)
        with (velo / 'recall-1m.log').open('w') as f:
            subprocess.run([sys.executable, str(HERE / 'recall.py'), '--tokens', '1048384',
                            '--fixed-repetitions', '40319', '--port', '19311', '--model', 'velo-yarn-audit',
                            '--out', str(velo / 'recall-1m.json')], stdout=f, stderr=subprocess.STDOUT, check=True)
        r = json.loads((velo / 'recall-1m.json').read_text())
        print('Candidate 1M:', r['seconds'], r['pass'], flush=True)
        full_cache('http://127.0.0.1:19311', 'velo-yarn-audit', velo)
        schema_probe('http://127.0.0.1:19311', 'velo-yarn-audit', velo)
        (a.root / 'completed.json').write_text(json.dumps({'wall': time.time()}))
    finally:
        if trial is not None:
            (velo / 'stop').touch()
            try:
                trial.wait(timeout=40)
            except subprocess.TimeoutExpired:
                trial.terminate()
                raise
        if log:
            log.close()


if __name__ == '__main__':
    main()
