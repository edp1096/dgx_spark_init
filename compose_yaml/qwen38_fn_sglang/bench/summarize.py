#!/usr/bin/env python3
"""Summarize a completed recheck without changing its source measurements."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    modes = {mode: json.loads((root / (mode + '.json')).read_text()) for mode in ['off', 'ko64k']}
    result = {'cases': {}, 'aggregate': {}, 'checks': {}, 'telemetry': {}}
    for name in dict.fromkeys(row['name'] for row in modes['off']['rows']):
        case = {}
        for mode, data in modes.items():
            rows = [row for row in data['rows'] if row['name'] == name]
            speeds = [row['decode_tok_s'] for row in rows]
            case[mode] = {'median_tok_s': statistics.median(speeds), 'min_tok_s': min(speeds), 'max_tok_s': max(speeds), 'samples': len(rows), 'outputs': [row['usage']['completion_tokens'] for row in rows]}
        case['change_percent'] = (case['ko64k']['median_tok_s']/case['off']['median_tok_s']-1)*100
        result['cases'][name] = case
    for mode, data in modes.items():
        rows = data['rows']
        result['aggregate'][mode] = {'tokens': sum(row['usage']['completion_tokens'] for row in rows), 'wall_seconds': sum(row['elapsed_s'] for row in rows)}
        total = result['aggregate'][mode]
        total['tok_s'] = total['tokens']/total['wall_seconds']
        result['checks'][mode] = [{'name': check['name'], 'passed': check['passed']} for check in data['checks']]
    telemetry = [json.loads(line) for line in (root/'telemetry.jsonl').read_text().splitlines()]
    for mode in modes:
        samples = [r for r in telemetry if r['phase'].startswith(mode+':')]
        temperatures = [float(r['gpu'].split(',')[0]) for r in samples if r.get('gpu')]
        result['telemetry'][mode] = {'min_available_gib': min(r['available_gib'] for r in samples), 'max_gpu_celsius': max(temperatures), 'samples': len(samples)}
    result['boot_ids'] = sorted(set(r['boot_id'] for r in telemetry))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), ensure_ascii=False, indent=2))
