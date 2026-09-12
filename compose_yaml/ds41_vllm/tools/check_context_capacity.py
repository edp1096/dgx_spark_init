"""Verify a long prefill with the real 8192-token output allowance."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import time
import urllib.request
from pathlib import Path

base = 'http://127.0.0.1:8010'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=(Path(__file__).resolve().parents[1] / 'results') / 'context-capacity-validation.json')
output = parser.parse_args().output
marker = 'spark-amethyst-642'


def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=600)


def messages(repetitions):
    half = repetitions // 2
    return [{'role': 'user', 'content':
        'Read the text and return only the exact validation marker.\n'
        + 'alpha beta gamma delta.\n' * half
        + '\nValidation marker: ' + marker + '\n'
        + 'alpha beta gamma delta.\n' * (repetitions - half)
        + '\nReturn only the exact validation marker from the middle of the text.'}]


repetitions = 9000
for _ in range(4):
    conversation = messages(repetitions)
    tokenized = json.load(post('/tokenize', {
        'model': 'deepseek-v4.1-flash', 'messages': conversation,
        'chat_template_kwargs': {'thinking': False}}))
    count = tokenized['count']
    if 54000 <= count <= 56500:
        break
    repetitions = int(repetitions * 55000 / count)
else:
    raise AssertionError(f'Could not size input: {count}')
assert count + 8192 < 65536
print(json.dumps({'input_tokens_estimate': count, 'output_allowance': 8192}), flush=True)
start = time.monotonic()
result = {'requested_output_tokens': 8192, 'tokenizer_count': count,
          'text': '', 'usage': None, 'finish_reason': None, 'ttft_seconds': None}
with post('/v1/chat/completions', {
    'model': 'deepseek-v4.1-flash', 'messages': conversation,
    'max_completion_tokens': 8192, 'temperature': 0,
    'chat_template_kwargs': {'thinking': False}, 'stream': True,
    'stream_options': {'include_usage': True}}) as response:
    result['status'] = response.status
    for line in response:
        if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
            continue
        event = json.loads(line[6:])
        assert not event.get('error'), event
        if event.get('usage'):
            result['usage'] = event['usage']
        if event.get('metrics'):
            result['metrics'] = event['metrics']
        for choice in event.get('choices', []):
            content = choice.get('delta', {}).get('content') or ''
            if content and result['ttft_seconds'] is None:
                result['ttft_seconds'] = time.monotonic() - start
            result['text'] += content
            if choice.get('finish_reason'):
                result['finish_reason'] = choice['finish_reason']
result['total_seconds'] = time.monotonic() - start
output.write_text(json.dumps(result, indent=2) + '\n')
assert result['text'].strip() == marker and result['finish_reason'] == 'stop'
assert result['usage']['prompt_tokens'] >= 54000
print(json.dumps(result), flush=True)
print('CONTEXT_CAPACITY_VALIDATION_PASS', flush=True)
