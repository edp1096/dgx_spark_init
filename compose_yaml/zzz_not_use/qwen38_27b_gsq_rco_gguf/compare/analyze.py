#!/usr/bin/env python3
"""Summarize measurements; factual prose still requires manual review."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

def structured(text):
    return json.loads(text.replace('```json', '').replace('```', '').strip())

def code_check(text):
    if '```' in text:
        text = text.split('```', 2)[1]
        if text.startswith('python'):
            text = text[6:]
    # Execute generated code in a disposable container with no mounts/network/GPU.
    checks = '''
import copy
examples = [([], []), ([[1,3],[2,6],[8,10],[15,18]], [[1,6],[8,10],[15,18]]), ([[2,3],[1,2]], [[1,3]]), ([[4,5],[-3,-1],[-2,0]], [[-3,0],[4,5]]), ([[1,4],[1,4],[2,3]], [[1,4]]), ([[0,0],[0,1]], [[0,1]]), ([[5,6],[1,2],[3,4]], [[1,2],[3,4],[5,6]])]
for value, expected in examples:
    original = copy.deepcopy(value)
    actual = merge_intervals(value)
    assert actual == expected, (actual, expected)
    assert value == original, 'input mutated'
print('PASS 7 cases including input preservation')
'''
    name = 'qwen27-code-check'
    try:
        result = subprocess.run(['docker', 'run', '--rm', '--name', name, '-i', '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '32', '--memory', '256m', '--cpus', '1', '--user', '65534:65534', '--entrypoint', 'python3', 'qwen38-27b-gsq-rco-gguf-tools:e71b805', '-I', '-'], input=text + '\n' + checks, text=True, capture_output=True, timeout=20)
        return {'passed': result.returncode == 0 and result.stdout.rstrip().endswith('PASS 7 cases including input preservation'), 'output': result.stdout + result.stderr}
    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
        return {'passed': False, 'output': 'timeout'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summary = {}
    for path in sorted(args.directory.glob('*/report.json')):
        report = json.loads(path.read_text())
        rows = report['benchmark']
        record = {'resources': report.get('resources'), 'quality': {}}
        if rows:
            record['client_tokens_per_second'] = sum(r['completion_tokens'] for r in rows) / sum(r['elapsed_seconds'] for r in rows)
            record['per_case_tps'] = {case: sum(r['completion_tokens'] for r in rows if r['case'] == case) / sum(r['elapsed_seconds'] for r in rows if r['case'] == case) for case in sorted({r['case'] for r in rows})}
            ttft = [r['ttft_seconds'] for r in report['stream'] if r['ttft_seconds'] is not None]
            record['warm_ttft_median_seconds'] = statistics.median(ttft) if ttft else None
            timings = [r['timings'] for r in rows if r.get('timings')]
            if timings:
                record['native_decode_tokens_per_second'] = sum(t['predicted_n'] for t in timings) / (sum(t['predicted_ms'] for t in timings) / 1000)
        for name, item in report['quality'].items():
            if name == 'vision' and report['label'].startswith('exl3'):
                content = (item.get('message') or {}).get('content') or ''
                try:
                    value = structured(content)
                    match = value.get('left', '').lower() == 'red' and value.get('right', '').lower() == 'blue'
                except (ValueError, TypeError, AttributeError):
                    match = False
                record['quality'][name] = {'passed': None, 'supported': False, 'raw_color_match': match, 'content': content, 'error': item.get('error'), 'reason': 'Pinned serve_openai.py sends only tokenizer chat-template IDs to Job; it has no image preprocessing or image embedding path.'}
                continue
            if 'error' in item:
                record['quality'][name] = {'passed': False, 'error': item['error']}
                continue
            message = item['message']
            content = message.get('content') or ''
            passed = None
            try:
                if name == 'korean':
                    passed = '16' in content and any('\uac00' <= c <= '\ud7a3' for c in content)
                elif name == 'tool-call':
                    calls = message.get('tool_calls') or []
                    values = json.loads(calls[0]['function']['arguments'])
                    passed = len(calls) == 1 and calls[0]['function']['name'] == 'document_generate' and values['format'] == 'hwp' and values['title'] == '분기 실적' and any('매출 300' in p for p in values['paragraphs'])
                elif name == 'tool-negative':
                    passed = bool(content) and not message.get('tool_calls')
                elif name == 'vision':
                    value = structured(content)
                    passed = value.get('left', '').lower() == 'red' and value.get('right', '').lower() == 'blue'
                elif name.startswith('retrieval'):
                    passed = content.strip() == 'SPARK-2749'
                elif name == 'math-quality':
                    passed = content.strip() == '7/11'
                elif name == 'code-quality':
                    record['quality'][name] = code_check(content)
                    continue
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                passed = False
            record['quality'][name] = {'passed': passed, 'content': content, 'elapsed_seconds': item['elapsed_seconds'], 'usage': item.get('usage')}
        summary[report['label']] = record
    (args.directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    for label, value in summary.items():
        print(label, round(value.get('client_tokens_per_second', 0), 3), {k: v['passed'] for k, v in value['quality'].items()})

if __name__ == '__main__':
    main()
