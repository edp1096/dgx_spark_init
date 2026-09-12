"""SparkTalk-shaped SSE requests: auto/no-call, call, result round trip, reasoning."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
import time
import urllib.request
from pathlib import Path

base = 'http://127.0.0.1:8010'
output = (Path(__file__).resolve().parents[1] / 'results') / 'tool-calling-validation.json'
tool = {'type': 'function', 'function': {
    'name': 'add_numbers', 'description': 'Return the sum of two integers.',
    'parameters': {'type': 'object', 'properties': {
        'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
        'required': ['a', 'b'], 'additionalProperties': False}}}
results = []


def request(label, messages, thinking=False, choice='auto'):
    body = {'model': 'deepseek-v4.1-flash', 'messages': messages,
            'stream': True, 'temperature': 0, 'max_completion_tokens': 256,
            'separate_reasoning': True, 'stream_reasoning': True,
            'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'thinking': thinking},
            'tools': [tool], 'tool_choice': choice}
    if thinking:
        body['chat_template_kwargs']['reasoning_effort'] = 'low'
    req = urllib.request.Request(base + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    result = {'label': label, 'content': '', 'reasoning': '', 'tool_calls': [],
              'finish_reason': None, 'usage': None, 'events': []}
    calls = {}
    done = False
    with urllib.request.urlopen(req, timeout=300) as response:
        result['status'] = response.status
        for line in response:
            if not line.startswith(b'data: '):
                continue
            if line.strip() == b'data: [DONE]':
                done = True
                break
            event = json.loads(line[6:])
            assert not event.get('error'), event
            result['events'].append(event)
            if event.get('usage'):
                result['usage'] = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                result['content'] += delta.get('content') or ''
                result['reasoning'] += delta.get('reasoning_content') or delta.get('reasoning') or ''
                if choice.get('finish_reason'):
                    result['finish_reason'] = choice['finish_reason']
                for part in delta.get('tool_calls') or []:
                    call = calls.setdefault(part['index'], {
                        'id': '', 'type': 'function',
                        'function': {'name': '', 'arguments': ''}})
                    if part.get('id'):
                        call['id'] = part['id']
                    fn = part.get('function') or {}
                    call['function']['name'] += fn.get('name') or ''
                    call['function']['arguments'] += fn.get('arguments') or ''
    result['tool_calls'] = [calls[i] for i in sorted(calls)]
    result['total_seconds'] = time.monotonic() - start
    results.append(result)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    assert done and result['usage'], result
    print(json.dumps({k: v for k, v in result.items() if k != 'events'}, ensure_ascii=False), flush=True)
    return result


for _ in range(360):
    try:
        urllib.request.urlopen(base + '/health', timeout=2)
        break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('API not ready')

identity = request('identity_with_auto_tools', [
    {'role': 'user', 'content': '넌 누구냐? 한국어 한 문장으로 답해.'}])
assert identity['content'].strip() and not identity['tool_calls']
assert identity['finish_reason'] == 'stop'

messages = [
    {'role': 'system', 'content': 'For addition, call add_numbers. After its result, '
     'answer with the result number only. Do not call a tool again after receiving its result.'},
    {'role': 'user', 'content': 'Use add_numbers with a=17 and b=25.'}]
called = request('auto_tool_call', messages)
assert called['finish_reason'] == 'tool_calls' and len(called['tool_calls']) == 1
call = called['tool_calls'][0]
assert call['id'] and call['function']['name'] == 'add_numbers'
args = json.loads(call['function']['arguments'])
assert args == {'a': 17, 'b': 25}
# Execute only this bounded local arithmetic fixture; no real application tools.
messages += [{'role': 'assistant', 'content': called['content'] or None,
              'tool_calls': called['tool_calls']},
             {'role': 'tool', 'tool_call_id': call['id'],
              'content': json.dumps({'result': args['a'] + args['b']})}]
finished = request('tool_result_round_trip', messages)
assert finished['content'].strip() == '42' and not finished['tool_calls']
assert finished['finish_reason'] == 'stop'

reasoned = request('reasoning_separation', [{'role': 'user',
    'content': 'What is 2 multiplied by 3? Think briefly and answer only the number.'}],
    thinking=True, choice='none')
assert reasoned['content'].strip() == '6' and reasoned['reasoning'].strip()
assert not reasoned['tool_calls'] and reasoned['finish_reason'] == 'stop'
print('TOOL_CALLING_VALIDATION_PASS', flush=True)
