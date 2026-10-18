"""GPU qualification, deterministic model comparisons and host UMA telemetry."""
import argparse
import base64
import json
import math
from pathlib import Path
import random
import re
import statistics
import struct
import subprocess
import threading
import time
import zlib

import requests

from download import atomic
from launch import launch, IMAGE


def meminfo():
    wanted = {'MemTotal', 'MemFree', 'MemAvailable', 'SwapTotal', 'SwapFree', 'Cached'}
    return {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.split(':')[0] in wanted}


class Monitor:
    def __init__(self, output):
        self.output, self.phase, self.rows = output, 'before_load', []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with (self.output / 'memory.jsonl').open('w') as log:
            while not self.done.is_set():
                row = {'time': time.time(), 'phase': self.phase, **meminfo()}
                self.rows.append(row)
                log.write(json.dumps(row) + '\n')
                log.flush()
                self.done.wait(1)

    def start(self):
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join()
        before, last = self.rows[0], self.rows[-1]
        steady = [r for r in self.rows if r['phase'] == 'ready_idle']
        summary = {'before': before, 'last': last,
            'min_available_gib': min(r['MemAvailable'] for r in self.rows) / 2**30,
            'min_free_gib': min(r['MemFree'] for r in self.rows) / 2**30,
            'peak_used_increase_gib': (before['MemAvailable'] - min(r['MemAvailable'] for r in self.rows)) / 2**30,
            'steady_used_increase_gib': (before['MemAvailable'] - statistics.median(r['MemAvailable'] for r in steady)) / 2**30 if steady else None,
            'peak_swap_gib': max(r['SwapTotal'] - r['SwapFree'] for r in self.rows) / 2**30,
            'swap_delta_gib': ((last['SwapTotal'] - last['SwapFree']) - (before['SwapTotal'] - before['SwapFree'])) / 2**30}
        atomic(self.output / 'memory-summary.json', summary)
        return summary


def metrics(session, base):
    response = session.get(base + '/metrics', timeout=15)
    response.raise_for_status()
    values = {}
    for line in response.text.splitlines():
        m = re.fullmatch(r'([A-Za-z_:][A-Za-z0-9_:]*)(?:\{.*\})?\s+([-+\d.eE]+)', line)
        if m:
            values[m[1]] = values.get(m[1], 0.) + float(m[2])
    return values


def streamed(session, base, payload, timeout=600):
    start, first = time.monotonic(), None
    text, reasoning, calls, usage, finish = '', '', {}, {}, None
    with session.post(base + '/v1/chat/completions', json=payload, stream=True, timeout=(20, timeout)) as response:
        if response.status_code != 200:
            raise RuntimeError(f'Completion HTTP {response.status_code}: {response.text[:2000]}')
        for line in response.iter_lines():
            if not line.startswith(b'data: '):
                continue
            if line == b'data: [DONE]':
                break
            event = json.loads(line[6:])
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                finish = choice.get('finish_reason') or finish
                d = choice.get('delta', {})
                content = d.get('content') or ''
                thought = d.get('reasoning_content') or d.get('reasoning') or ''
                tool = d.get('tool_calls') or []
                if first is None and (content or thought or tool):
                    first = time.monotonic()
                text += content
                reasoning += thought
                for part in tool:
                    i = part.get('index', 0)
                    call = calls.setdefault(i, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                    if part.get('id'):
                        call['id'] = part['id']
                    f = part.get('function', {})
                    call['function']['name'] += f.get('name') or ''
                    call['function']['arguments'] += f.get('arguments') or ''
    elapsed = time.monotonic() - start
    ttft = first - start if first else None
    tokens = usage.get('completion_tokens', 0)
    return {'text': text, 'reasoning': reasoning, 'tool_calls': list(calls.values()),
        'usage': usage, 'finish_reason': finish, 'seconds': elapsed, 'ttft_s': ttft,
        'stream_tg_estimate': max(0, tokens - 1) / (elapsed - ttft) if ttft is not None and elapsed > ttft else None}


def fixture_image():
    def chunk(kind, value):
        return struct.pack('>I', len(value)) + kind + value + struct.pack('>I', zlib.crc32(kind + value) & 0xffffffff)
    w, h = 224, 224
    row = bytes([255, 0, 0]) * (w // 2) + bytes([0, 0, 255]) * (w // 2)
    raw = b''.join(b'\0' + row for _ in range(h))
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


TOOL = {'type': 'function', 'function': {'name': 'weather_lookup',
    'description': 'Look up the current weather in the requested city.',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'enum': ['Seoul']},
        'unit': {'type': 'string', 'enum': ['celsius']}},
        'required': ['city', 'unit'], 'additionalProperties': False}}}


def cases():
    yield 'warmup', 'Reply with READY.', 16, {}, 'READY'
    yield 'arithmetic', 'Compute 137 * 29. Reply with only the integer.', 32, {}, '3973'
    yield 'reasoning', '3x+7=52일 때 2x²-5를 계산해라. 풀이를 검토한 뒤 최종 답변에는 정수만 써라.', 2048, {
        'reasoning_effort': 'medium', 'chat_template_kwargs': {'enable_thinking': True}}, '445'
    schema = {'type': 'object', 'properties': {'name': {'type': 'string', 'enum': ['테스트']},
        'count': {'type': 'integer', 'enum': [7]}}, 'required': ['name', 'count'], 'additionalProperties': False}
    yield 'json_schema', '이름은 테스트, 개수는 7인 JSON 객체를 반환해라.', 100, {
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'record', 'strict': True, 'schema': schema}}}, None
    yield 'korean', '한국의 궁궐에서 정전과 편전이 각각 어떤 역할을 했는지 한국어 세 문단으로 설명해라.', 600, {}, None
    yield 'translation', '다음 글을 한국어로 정확하게 번역해라. 번역문만 답해라.\nDinah was sitting by the fire. Alice looked at her cat and said, "She is asleep, but I am not tired." No one else was in the room.', 400, {}, None
    yield 'fiction', '야간 경비원과 그림을 훔치러 온 도둑이 미술관에서 마주치는 소설을 한국어 800~1,200자로 써라. 두 인물의 대화와 반전 결말을 포함하고 작품만 출력해라.', 2000, {}, None
    yield 'coding_unique', 'Write only Python code defining unique_ordered(items). Remove duplicates while preserving first appearance. Items are hashable. Do not mutate the input. No markdown.', 600, {}, None
    yield 'coding_intervals', 'Write only Python code defining merge_intervals(intervals). Each interval is a pair of integers [start,end], start<=end. Merge overlapping or touching intervals, return a sorted list of [start,end] lists, and do not mutate the input. Handle empty input. No markdown.', 1000, {}, None
    yield 'tool', '서울 날씨를 제공된 도구로 확인해라. city는 Seoul, unit은 celsius를 사용해라.', 300, {'tools': [TOOL], 'tool_choice': 'auto'}, None
    yield 'tool_literal', '실제로 도구를 호출하지 마라. weather_lookup이라는 함수 이름이 들어간 XML 도구 호출 예시를 코드 블록 안에 적고, 예시라는 설명을 덧붙여라.', 500, {'tools': [TOOL], 'tool_choice': 'auto'}, None
    content = [{'type': 'text', 'text': 'Identify the left and right colors. Reply only as two lowercase English color names separated by a comma, left first.'},
               {'type': 'image_url', 'image_url': {'url': fixture_image()}}]
    yield 'vision', content, 80, {}, 'red,blue'
    for i in range(3):
        yield 'speed_' + str(i), f'실험 {i}. 한국 궁궐의 건축과 공간 배치, 자연과의 관계를 구체적인 사례를 들어 2,000자 이상 자세하게 설명해라.', 512, {}, None


def code_test(output, name, text):
    code = text.strip()
    if code.startswith('```'):
        code = re.sub(r'^```(?:python)?\s*\n', '', code)
        code = re.sub(r'\n```\s*$', '', code)
    directory = output / name
    directory.mkdir(exist_ok=True)
    (directory / 'solution.py').write_text(code)
    if name == 'coding_unique':
        body = '''from solution import unique_ordered
cases=[([],[]),([1,2,1,3,2],[1,2,3]),(["a","b","a"],["a","b"]),([(1,2),(1,2),(2,1)],[(1,2),(2,1)])]
for value,want in cases:
 before=value.copy()
 assert unique_ordered(value)==want
 assert value==before
'''
    else:
        body = '''from solution import merge_intervals
import copy
cases=[([],[]),([[1,3],[2,6],[8,10],[10,12]],[[1,6],[8,12]]),([[5,7],[-3,-1],[-1,2]], [[-3,2],[5,7]]),([[1,10],[2,3],[4,8]],[[1,10]]),([[2,2],[1,2]],[[1,2]])]
for value,want in cases:
 before=copy.deepcopy(value)
 assert merge_intervals(value)==want
 assert value==before
'''
    (directory / 'test.py').write_text(body + '\nprint("ALL_CASES_PASSED")\n')
    container = 'fp4-code-' + name.replace('_', '-') + '-' + str(time.time_ns())
    cmd = ['docker', 'run', '--rm', '--name', container, '--network', 'none', '--read-only',
        '--cap-drop=ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64',
        '--memory', '512m', '--cpus', '1', '--user', '65534:65534',
        '-e', 'PYTHONOPTIMIZE=0', '--entrypoint', 'python3', '-v', f'{directory}:/cases:ro', IMAGE, '/cases/test.py']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        return {'passed': result.returncode == 0 and result.stdout.strip() == 'ALL_CASES_PASSED',
                'exit_code': result.returncode, 'stdout': result.stdout[-2000:], 'stderr': result.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {'passed': False, 'error': 'Code execution timed out'}
    finally:
        subprocess.run(['docker', 'rm', '-f', container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def long_prompt(repeats):
    words = ('river stone cloud copper forest window garden bridge paper clock silver amber blue green red '
             'white black yellow orange purple house door table chair room floor wall roof road path hill '
             'mountain valley lake sea beach island field grass tree leaf branch root flower seed fruit '
             'apple pear peach grape bird fish horse sheep lion tiger bear wolf fox deer sun moon star '
             'light shadow morning evening night spring winter summer autumn rain snow wind storm water '
             'fire earth sand book letter page pencil bottle basket box bowl plate cup shirt coat shoe hat').split()
    rng = random.Random(20261016)
    segments = [''.join(' '.join(rng.choices(words, k=8)) + '.\n' for _ in range(repeats)) for _ in range(3)]
    return ('Read this archive and remember its two named fields.\n' + segments[0] +
        '\nFIELD_ALPHA = MAPLE\n' + segments[1] + '\nFIELD_BETA = COMET\n' + segments[2] +
        '\nReturn only the values of FIELD_ALPHA and FIELD_BETA in that order, as two uppercase words separated by a comma.')


def run(root, model_path, label, context, port, utilization, run_long):
    output = root / 'runtime' / label
    output.mkdir(parents=True, exist_ok=True)
    monitor = Monitor(output)
    monitor.start()
    session = requests.Session()
    session.trust_env = False
    base = f'http://127.0.0.1:{port}'
    record = None
    results = []
    summary = {'status': 'starting', 'label': label, 'context': context, 'started': time.time(), 'results': results}
    def save():
        atomic(output / 'qualification.json', summary)
    save()
    try:
        monitor.phase = 'loading'
        record = launch(root, model_path, label, context, port, utilization)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            state = json.loads(subprocess.check_output(['docker', 'inspect', record['container']]))[0]['State']
            if not state['Running']:
                raise RuntimeError('GPU server exited during startup: ' + str(state['ExitCode']))
            try:
                response = session.get(base + '/v1/models', timeout=3)
                if response.status_code == 200:
                    summary['models'] = response.json()
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            raise RuntimeError('GPU server startup timed out')
        summary['startup_seconds'] = time.time() - summary['started']
        summary['status'] = 'running'
        save()
        served = record['served_model']
        print('GPU SERVER READY', label, summary['startup_seconds'], flush=True)
        monitor.phase = 'ready_idle'
        time.sleep(5)
        initial_metrics = metrics(session, base)

        def call(name, content, maximum, extra=None, timeout=600):
            monitor.phase = name
            payload = {'model': served, 'messages': [{'role': 'user', 'content': content}],
                'temperature': 0, 'seed': 42, 'max_tokens': maximum,
                'stream': True, 'stream_options': {'include_usage': True},
                'chat_template_kwargs': {'enable_thinking': False}, **(extra or {})}
            before = metrics(session, base)
            row = streamed(session, base, payload, timeout)
            after = metrics(session, base)
            delta = {k: v - before.get(k, 0.) for k, v in after.items()}
            row.update(name=name, metric_delta={k: v for k, v in delta.items() if 'prefill' in k or 'decode' in k or 'tokens_total' in k})
            prefill = delta.get('vllm:request_prefill_time_seconds_sum', 0.)
            cached = (row['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0)
            row['prefill_tokens_per_second'] = (row['usage'].get('prompt_tokens', 0) - cached) / prefill if prefill > 0 else None
            results.append(row)
            save()
            return row

        for name, content, maximum, extra, expected in cases():
            try:
                row = call(name, content, maximum, extra)
                if expected:
                    row['passed'] = row['text'].strip().replace(' ', '').rstrip('.').lower() == expected.lower()
                if name == 'json_schema':
                    row['passed'] = json.loads(row['text']) == {'name': '테스트', 'count': 7}
                if name == 'reasoning':
                    row['passed'] = row['passed'] and bool(row['reasoning'].strip())
                if name.startswith('coding_'):
                    row['code_test'] = code_test(output, name, row['text'])
                    row['passed'] = row['code_test']['passed']
                if name == 'tool':
                    calls = row['tool_calls']
                    row['passed'] = (len(calls) == 1 and calls[0]['function']['name'] == 'weather_lookup'
                        and json.loads(calls[0]['function']['arguments']) == {'city': 'Seoul', 'unit': 'celsius'})
                    if row['passed']:
                        follow = call('tool_result', '', 256, {'tools': [TOOL], 'tool_choice': 'auto', 'messages': [
                            {'role': 'user', 'content': content},
                            {'role': 'assistant', 'content': row['text'] or None, 'tool_calls': calls},
                            {'role': 'tool', 'tool_call_id': calls[0]['id'],
                             'content': json.dumps({'city': 'Seoul', 'temperature_c': 18, 'condition': 'clear'})}]})
                        follow['passed'] = '18' in follow['text'] and not follow['tool_calls']
                if name == 'tool_literal':
                    row['passed'] = not row['tool_calls'] and bool(row['text'].strip())
                if name in ('korean', 'translation', 'fiction'):
                    row['hangul_characters'] = sum('\uac00' <= ch <= '\ud7a3' for ch in row['text'])
                    row['requires_semantic_review'] = True
                    row['passed'] = row['finish_reason'] == 'stop' and row['hangul_characters'] > (350 if name == 'fiction' else 20)
            except Exception as exc:
                row = {'name': name, 'passed': False, 'error': str(exc)[:2000]}
                results.append(row)
            save()
            print('CASE', label, name, row.get('passed'), round(row.get('seconds', 0), 2), flush=True)
        if run_long:
            monitor.phase = 'long_tokenization'
            target, repeats = context - 512, max(1, (context - 600) // 30)
            count = 0
            for _ in range(8):
                prompt = long_prompt(repeats)
                response = session.post(base + '/tokenize', json={'model': served,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'chat_template_kwargs': {'enable_thinking': False}, 'add_generation_prompt': True}, timeout=(20, 300))
                response.raise_for_status()
                count = response.json()['count']
                if 0 <= target - count < 96:
                    break
                per_repeat = max(1., (count - 64) / repeats)
                repeats = max(1, repeats + int((target - count) / per_repeat) - int(count > target))
            if not 0 <= target - count < 96:
                raise RuntimeError('Could not fit the actual long prompt: ' + str(count))
            atomic(output / 'long-input-plan.json', {'actual_tokenized_tokens': count, 'target': target,
                'repeats_per_segment': repeats, 'diverse_seed': 20261016})
            print('LONG INPUT BEGIN', label, count, flush=True)
            row = call('long_retrieval', prompt, 64, timeout=7200)
            row['expected_prompt_tokens'] = count
            row['passed'] = row['text'].strip().replace(' ', '') == 'MAPLE,COMET' and row['usage'].get('prompt_tokens') == count
            save()
            print('LONG INPUT END', label, row['passed'], row['seconds'], flush=True)
            row = call('post_long', '17과 25를 더한 값을 숫자로만 답해라.', 32)
            row['passed'] = row['text'].strip() == '42'
        final_metrics = metrics(session, base)
        summary['speculative_metrics_delta'] = {k: v - initial_metrics.get(k, 0.) for k, v in final_metrics.items() if 'spec_decode' in k}
        summary['status'] = 'passed' if all(r.get('passed', True) and not r.get('error') for r in results) else 'checks_failed'
        summary['finished'] = time.time()
        save()
    except Exception as exc:
        summary.update(status='failed', error=str(exc)[:3000], finished=time.time())
        save()
        raise
    finally:
        if record:
            logs = subprocess.run(['docker', 'logs', record['container']], capture_output=True, text=True)
            (output / 'server.log').write_text(logs.stdout + logs.stderr)
            monitor.phase = 'stopping'
            subprocess.run(['docker', 'stop', '-t', '30', record['container']], capture_output=True)
            state = json.loads(subprocess.check_output(['docker', 'inspect', record['container']]))[0]
            atomic(output / 'container-state.json', {'state': state['State'], 'image': state['Image']})
        monitor.phase = 'stopped'
        time.sleep(2)
        summary['memory'] = monitor.stop()
        save()
        session.close()
    print('QUALIFICATION FINISHED', label, summary['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--label', choices=['base', 'candidate', 'base-native', 'candidate-native'], required=True)
    parser.add_argument('--context', type=int, choices=[262144, 1048576], default=1048576)
    parser.add_argument('--port', type=int, default=30136)
    parser.add_argument('--utilization', type=float, default=.8)
    parser.add_argument('--long', action='store_true')
    a = parser.parse_args()
    run(a.root, a.model, a.label, a.context, a.port, a.utilization, a.long)
