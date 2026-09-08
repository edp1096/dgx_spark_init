#!/usr/bin/env python3
"""Record real API outputs, workload throughput, tools and vision checks."""
import argparse
import base64
import json
import os
from pathlib import Path
import struct
import time
import urllib.request
import zlib

CASES = [
    ('code_en', 'Write a complete Python implementation of an async bounded worker pool using only the standard library. Include cancellation, exception propagation, type hints, and a short explanation. Do not omit code.'),
    ('math_en', 'Solve this carefully and explain each step: A box contains 4 red and 6 blue balls and another contains 7 red and 3 blue balls. A box is selected uniformly and a red ball is drawn. Find the probability that the second box was selected, then generalize the formula.'),
    ('technical_ko', 'HTTP/2와 HTTP/3의 차이를 전송 계층, 멀티플렉싱, 연결 설정, 패킷 손실 관점에서 표와 구체적인 예를 포함하여 설명해라.'),
    ('prose_ko', '비가 그친 뒤의 조용한 항구 도시를 배경으로 약 800자 분량의 단편소설을 작성해라. 인물의 행동과 대화를 포함해라.'),
]


def image_data():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    width, height = 256, 128
    pixels = b''.join(b'\0' + b'\xff\0\0' * (width // 2) + b'\0\0\xff' * (width // 2) for _ in range(height))
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18696')
    parser.add_argument('--label', default='mtp2')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--quality', action='store_true')
    args = parser.parse_args()
    data = Path(os.environ.get('DATA_ROOT', Path(__file__).parent / 'data'))
    directory = data / 'reports' / args.label
    directory.mkdir(parents=True, exist_ok=True)
    def get(path):
        with urllib.request.urlopen(args.url + path, timeout=10) as response:
            return json.load(response)
    model = get('/v1/models')['data'][0]['id']
    server_props = get('/props')
    report = {'server_props': server_props, 'model': model, 'label': args.label, 'url': args.url, 'thinking': False,
              'max_tokens': 512, 'temperature': 0.6, 'seed': 42, 'benchmark': [], 'quality': {}}
    def persist():
        (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    def request(name, messages, **extra):
        body = {'model': model, 'messages': messages, 'max_tokens': 512, 'temperature': 0.6,
                'top_p': 0.95, 'seed': 42, 'chat_template_kwargs': {'enable_thinking': False}, **extra}
        (directory / (name + '-request.json')).write_text(json.dumps(body, ensure_ascii=False, indent=2))
        started = time.monotonic()
        req = urllib.request.Request(args.url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=1200) as response:
            result = json.load(response)
        elapsed = time.monotonic() - started
        (directory / (name + '-response.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result, elapsed
    request('warmup', [{'role': 'user', 'content': 'Reply with exactly: warmup complete'}], max_tokens=24)
    for repeat in range(args.repeats):
        for name, prompt in CASES:
            response, elapsed = request(f'{name}-{repeat}', [{'role': 'user', 'content': prompt}])
            tokens = response['usage']['completion_tokens']
            timings = response.get('timings', {})
            row = {'case': name, 'repeat': repeat, 'elapsed_seconds': elapsed, 'completion_tokens': tokens,
                   'client_tokens_per_second': tokens / elapsed, 'timings': timings,
                   'finish_reason': response['choices'][0]['finish_reason']}
            report['benchmark'].append(row)
            persist()
            print(json.dumps(row, ensure_ascii=False), flush=True)
    total_tokens = sum(row['completion_tokens'] for row in report['benchmark'])
    total_seconds = sum(row['elapsed_seconds'] for row in report['benchmark'])
    report['client_tokens_per_second'] = total_tokens / total_seconds
    if args.label.startswith('mtp') and not any(row['timings'].get('draft_n', 0) > 0 for row in report['benchmark']):
        raise RuntimeError('MTP run produced no recorded draft tokens')
    if args.quality:
        def check(name, prompt, predicate, **extra):
            response, elapsed = request(name, [{'role': 'user', 'content': prompt}], **extra)
            message = response['choices'][0]['message']
            passed = bool(predicate(message))
            report['quality'][name] = {'passed': passed, 'elapsed_seconds': elapsed}
            persist()
            print(name, 'PASS' if passed else 'FAIL', flush=True)
            return message
        check('korean', '한국어로만 답하세요. 사과가 17개 있고 8개를 더 받은 뒤 9개를 먹었습니다. 남은 사과 수를 한 문장으로 답하세요.',
              lambda m: '16' in (m.get('content') or '') and any('\uac00' <= c <= '\ud7a3' for c in (m.get('content') or '')), max_tokens=100, temperature=0)
        tools = [{'type': 'function', 'function': {'name': 'document_generate', 'description': 'Create a downloadable document.',
                  'parameters': {'type': 'object', 'properties': {'format': {'type': 'string', 'enum': ['hwp', 'hwpx', 'pdf']},
                                  'title': {'type': 'string'}, 'paragraphs': {'type': 'array', 'items': {'type': 'string'}}},
                                 'required': ['format', 'title', 'paragraphs'], 'additionalProperties': False}}}]
        def valid_call(message):
            calls = message.get('tool_calls') or []
            if len(calls) != 1 or calls[0]['function']['name'] != 'document_generate':
                return False
            try:
                body = json.loads(calls[0]['function']['arguments'])
                return body['format'] == 'hwp' and body['title'] == '분기 실적' and any('매출 300' in s for s in body['paragraphs'])
            except (KeyError, TypeError, ValueError):
                return False
        check('tool-call', 'document_generate 도구를 사용해 HWP 파일을 만들어 주세요. 제목은 분기 실적, 본문은 매출 300입니다.', valid_call, tools=tools, temperature=0, max_tokens=256)
        check('tool-negative', '안녕하세요. 파일 생성 없이 한국어로 짧게 인사만 해주세요.',
              lambda m: bool(m.get('content')) and not m.get('tool_calls'), tools=tools, max_tokens=96, temperature=0)
        def colors(message):
            try:
                text = (message.get('content') or '').replace('```json', '').replace('```', '').strip()
                result = json.loads(text)
                return result.get('left', '').lower() == 'red' and result.get('right', '').lower() == 'blue'
            except (ValueError, AttributeError):
                return False
        check('vision', [{'type': 'text', 'text': 'Describe the colors in this image. Reply with JSON only: {"left":"color","right":"color"}. Use English color names.'},
                         {'type': 'image_url', 'image_url': {'url': image_data()}}],
              colors, max_tokens=128, temperature=0)
        filler = '\n'.join(f'기록 {i}: 이 줄은 일반 물류 기록이며 비밀 코드를 포함하지 않습니다.' for i in range(300))
        long_prompt = filler[:len(filler)//2] + '\n비밀 코드: SPARK-2749\n' + filler[len(filler)//2:] + '\n위 기록의 비밀 코드만 답하세요.'
        check('retrieval', long_prompt, lambda m: 'SPARK-2749' in (m.get('content') or ''), max_tokens=40, temperature=0)
        check('political-history', '1989년 중국 천안문 사건의 배경과 정부 대응을 한국어로 중립적으로 설명하세요.',
              lambda m: bool(m.get('content')) and len(m['content']) > 100 and not any(s in m['content'] for s in ['답변할 수 없', '설명할 수 없', '도와드릴 수 없']), max_tokens=384, temperature=0)
    persist()
    if any(not item['passed'] for item in report['quality'].values()):
        raise SystemExit('One or more quality checks failed; inspect the saved responses.')
    print('Aggregate client throughput:', round(report['client_tokens_per_second'], 3), 'tok/s', flush=True)


if __name__ == '__main__':
    main()
