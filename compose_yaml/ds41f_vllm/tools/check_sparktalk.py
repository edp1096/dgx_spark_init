"""Exercise the running SparkTalk chat handler with its actual tools and budgets."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
import argparse
import time
import urllib.request
from pathlib import Path

base = 'http://127.0.0.1:8585'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=(Path(__file__).resolve().parents[1] / 'results') / 'sparktalk-context-validation.json')
parser.add_argument('--output-reserve', type=int, default=16384)
parser.add_argument('--marker')
parser.add_argument('--reasoning-effort')
args = parser.parse_args()
output = args.output


def api(path, body=None, method=None):
    req = urllib.request.Request(base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'}, method=method)
    return urllib.request.urlopen(req, timeout=600)


for _ in range(360):
    try:
        urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=2)
        break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('Model API not ready')

cfg = json.load(api('/api/config'))
assert cfg['model']['endpoint'].rstrip('/') == 'http://127.0.0.1:8010'
assert cfg['context']['output_reserve'] == args.output_reserve
session = json.load(api('/api/sessions', {'title': 'Temporary API connection validation'}))
result = {'session_id': session['id'], 'context_settings': cfg['context'],
          'events': [], 'content': '', 'done': False, 'temporary_session_deleted': False}
start = time.monotonic()
try:
    req = {'session_id': session['id'], 'content': '넌 누구냐?',
           'tools_enabled': True}
    if args.marker:
        req['content'] = f'검증용이다. 도구 사용이나 다른 설명 없이 {args.marker} 만 그대로 출력해.'
    if args.reasoning_effort:
        req['reasoning_effort'] = args.reasoning_effort
    kind = None
    with api('/api/chat', req) as response:
        result['http_status'] = response.status
        for line in response:
            if line.startswith(b'event: '):
                kind = line[7:].strip().decode()
            elif line.startswith(b'data: '):
                data = json.loads(line[6:])
                result['events'].append({'kind': kind, 'data': data})
                if kind == 'delta':
                    result['content'] += data.get('delta', '')
                if kind == 'done':
                    result['done'] = True
                if kind == 'error':
                    raise AssertionError(data)
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    assert result['done'] and result['content'].strip()
    if args.marker:
        assert result['content'].strip() == args.marker
    result['total_seconds'] = time.monotonic() - start
    print(json.dumps({k: v for k, v in result.items() if k != 'events'}, ensure_ascii=False), flush=True)
finally:
    # Delete only the isolated session created by this validation.
    with api('/api/sessions/' + session['id'], method='DELETE') as response:
        result['temporary_session_deleted'] = response.status == 204
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print('SPARKTALK_CHAT_VALIDATION_PASS', flush=True)
