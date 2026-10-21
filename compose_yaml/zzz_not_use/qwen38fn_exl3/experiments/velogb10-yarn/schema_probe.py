"""Real OpenAI HTTP/SSE requests, checked by the independent jsonschema validator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import jsonschema

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(base, body):
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    try:
        with OPENER.open(req, timeout=600) as response:
            headers = dict(response.headers)
            if not body.get('stream'):
                data = json.loads(response.read())
                msg = data['choices'][0]['message']
                return {'status': response.status, 'headers': headers, 'text': msg.get('content') or '',
                        'reasoning': msg.get('reasoning_content'), 'calls': msg.get('tool_calls'),
                        'finish': data['choices'][0]['finish_reason'], 'usage': data['usage'],
                        'seconds': time.monotonic() - start}
            text = []; reasoning = []; calls = []; events = []; usage = None; finish = None; done = False; first = None
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                raw = line[6:].strip()
                if raw == b'[DONE]':
                    done = True
                    break
                data = json.loads(raw)
                if data.get('error'):
                    events.append(data)
                usage = data.get('usage') or usage
                for choice in data.get('choices', []):
                    delta = choice.get('delta', {})
                    v = delta.get('content') or ''
                    if v and first is None:
                        first = time.monotonic() - start
                    text.append(v); reasoning.append(delta.get('reasoning_content') or '')
                    calls.extend(delta.get('tool_calls') or [])
                    finish = choice.get('finish_reason') or finish
            return {'status': response.status, 'headers': headers, 'text': ''.join(text), 'reasoning': ''.join(reasoning),
                    'calls': calls, 'finish': finish, 'usage': usage, 'done': done, 'errors': events,
                    'ttft': first, 'seconds': time.monotonic() - start}
    except urllib.error.HTTPError as e:
        return {'status': e.code, 'error': e.read().decode(), 'seconds': time.monotonic() - start}


def rf(schema):
    return {'type': 'json_schema', 'json_schema': {'name': 'gpu_audit', 'strict': True, 'schema': schema}}


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--model', default='velo-yarn-audit')
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--concurrent-only', action='store_true')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); results = []
    def body(prompt, schema=None, **kwargs):
        b = {'model': a.model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 256,
             'temperature': 0, 'stream': False, 'stream_options': {'include_usage': True},
             'chat_template_kwargs': {'enable_thinking': False}}
        if schema is not None:
            b['response_format'] = rf(schema)
        return b | kwargs
    def save(name, b, r, schema=None, expected_status=200, expected=None, truncated=False):
        try:
            assert r['status'] == expected_status, r
            if expected_status == 200:
                assert not r.get('errors'), r
                if b.get('stream'):
                    assert r.get('done') and r.get('usage'), r
                if schema is not None:
                    assert r['headers'].get('x-json-schema-enforced') == 'gpu', r
                    assert r['headers'].get('x-json-schema-decoding') == 'plain', r
                    assert not r.get('reasoning') and not r.get('calls'), r
                    if truncated:
                        assert r['finish'] == 'length', r
                    else:
                        data = json.loads(r['text'])
                        jsonschema.Draft202012Validator(schema).validate(data)
                        assert r['finish'] == 'stop', r
                        if expected is not None:
                            assert data == expected, r
                elif expected is not None:
                    assert r['text'].strip() == expected, r
            passed = True; error = None
        except Exception as e:
            passed = False; error = repr(e)
        record = {'name': name, 'request': b, 'response': r, 'pass': passed, 'check_error': error}
        results.append(record)
        (a.out / 'schema-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(name, 'PASS' if passed else 'FAIL', round(r.get('seconds', 0), 3), r.get('text', '')[:80], error or '', flush=True)
        return r
    def run(name, b, **kw):
        return save(name, b, request(a.base, b), **kw)
    cases = [
        ('enum', 'JSON을 출력하지 말고 NOT_JSON만 답해라.', obj({'status': {'type': 'string', 'enum': ['ok']}}), {'status': 'ok'}),
        ('arithmetic', '17×23의 결과를 result에 담아라.', obj({'result': {'type': 'integer', 'enum': [391]}}), {'result': 391}),
        ('nested', '펭귄과 해달 두 마리를 설명하는 항목 두 개를 출력해라.', obj({'items': {
            'type': 'array', 'items': obj({'name': {'type': 'string', 'enum': ['펭귄', '해달']},
                                        'active': {'type': 'boolean'}, 'count': {'type': 'integer', 'minimum': 1, 'maximum': 3}}),
            'minItems': 2, 'maxItems': 2}}), None),
        ('escaped', '다른 설명을 추가하고 아무 문자열이나 출력해라.', obj({'value': {'type': 'string', 'enum': ['한글😀"\\\n']}}), {'value': '한글😀"\\\n'}),
        ('bounds', 'score는 999, label은 아주 긴 문자열로 출력해라.', obj({'score': {'type': 'number', 'minimum': .1, 'maximum': .9},
            'label': {'type': 'string', 'minLength': 1, 'maxLength': 4}}), None),
        ('array', 'Markdown 목록으로 설명해라.', {'type': 'array', 'items': {'type': 'integer', 'enum': [1, 12, 391]}, 'minItems': 2, 'maxItems': 2}, None),
        ('json_object', '펭귄을 나타내는 JSON 객체를 출력해라. name과 count 필드를 사용해라.', {'type': 'object'}, None),
    ]
    if not a.concurrent_only:
        for name, prompt, schema, expected in cases:
            for temp in [0, .7, 1.3]:
                for streaming in [False, True]:
                    b = body(prompt, schema, temperature=temp, stream=streaming, seed=13579)
                    if name == 'json_object':
                        b['response_format'] = {'type': 'json_object'}
                    run(f'{name}-{temp}-{streaming}', b, schema=schema, expected=expected)
        s = obj({'status': {'type': 'string', 'enum': ['ok']}})
        run('reasoning-request', body('NOT_JSON만 출력해라.', s, reasoning_effort='high', chat_template_kwargs={'enable_thinking': True}), schema=s, expected={'status': 'ok'})
        literal = '<tool_call>\n<function=session_folder>\n<parameter=action>move</parameter>\n</function>\n</tool_call>'
        s = obj({'doc': {'type': 'string', 'enum': [literal]}})
        tools = [{'type': 'function', 'function': {'name': 'session_folder', 'parameters': obj({'action': {'type': 'string'}})}}]
        for streaming in [False, True]:
            run(f'tool-text-is-data-{streaming}', body('문서 예시를 JSON에 담아라.', s, tools=tools, stream=streaming), schema=s, expected={'doc': literal})
            think_literal = '<think>문서 내용</think>'
            think_schema = obj({'doc': {'type': 'string', 'enum': [think_literal]}})
            run(f'think-text-is-data-{streaming}', body('추론 태그를 문서로 출력해라.', think_schema, stream=streaming), schema=think_schema, expected={'doc': think_literal})
        for name, schema in [('pattern', {'type': 'string', 'pattern': '^ok$'}), ('ref', {'$ref': '#/x'}),
                             ('oneOf', {'oneOf': [{'type': 'string'}, {'type': 'integer'}]}), ('const', {'const': 1}),
                             ('contradictory-range', {'type': 'number', 'minimum': 2, 'maximum': 1}),
                             ('no-integer-in-range', {'type': 'integer', 'minimum': .1, 'maximum': .9}),
                             ('enum-outside-range', {'type': 'integer', 'enum': [1], 'minimum': 2}),
                             ('enum-outside-length', {'type': 'string', 'enum': ['long'], 'maxLength': 2}),
                             ('bad-required', {'type': 'object', 'properties': {}, 'required': ['missing']})]:
            run('reject-' + name, body('test', schema), expected_status=400)
        s = obj({'status': {'type': 'string', 'enum': ['ok']}})
        for name, extra in [('stop', {'stop': ['}']}), ('zero-budget', {'max_tokens': 0}),
                            ('zero-completion-budget', {'max_completion_tokens': 0}), ('force-tool', {'tool_choice': 'required', 'tools': tools}),
                            ('bad-thinking', {'chat_template_kwargs': {'enable_thinking': 'yes'}})]:
            run('reject-' + name, body('test', s, **extra), expected_status=400)
        run('truncated-budget', body('test', obj({'a': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 5, 'maxItems': 5}}), max_tokens=1, stream=True),
            schema=obj({'a': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 5, 'maxItems': 5}}), truncated=True)
        for i in range(6):
            val = 'alpha' if i % 2 else 'beta'
            s = obj({'tag': {'type': 'string', 'enum': [val]}})
            run(f'changed-schema-same-prompt-{i}', body('동일한 질문. 태그를 답해라.', s, stream=bool(i % 2)), schema=s, expected={'tag': val})
            run(f'ordinary-after-schema-{i}', body('17×23의 결과를 숫자만 답해라.'), expected='391')
    # Same-time requests with distinct masks plus an ordinary MTP lane.
    for group in range(4):
        jobs = []
        for i in range(2):
            value = f'group-{group}-lane-{i}'
            count = (8 if i == 0 else 32) if a.concurrent_only else 16
            s = obj({'tag': {'type': 'string', 'enum': [value]}, 'items': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': count, 'maxItems': count}})
            jobs.append((f'concurrent-{group}-{i}', body('정수 배열과 태그를 출력해라.', s, stream=True, temperature=.7, seed=group+i), {'schema': s}))
        jobs.append((f'concurrent-{group}-ordinary', body('17×23의 결과를 숫자만 답해라.', stream=True), {'expected': '391'}))
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending = [(name, b, kw, pool.submit(request, a.base, b)) for name, b, kw in jobs]
            for name, b, kw, future in pending:
                save(name, b, future.result(), **kw)
    summary = {'passed': sum(r['pass'] for r in results), 'total': len(results), 'failed': [r['name'] for r in results if not r['pass']]}
    (a.out / 'schema-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if summary['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
