#!/usr/bin/env python3
"""Summarize completed vocab_memory.py trials without touching a running server."""
import json
from pathlib import Path
import re
import sys


def summarize(root):
    if not (root / 'complete.json').exists():
        raise ValueError('trial set is incomplete')
    run = json.loads((root / 'run.json').read_text())
    result = {'settings': {k: run[k] for k in ['context_length', 'kv_tokens', 'mamba_slots', 'min_available_gib', 'rounds']}, 'rows': []}
    for mode in ['ko64k', 'ko128k', 'full']:
        mem = json.loads((root / f'{mode}-memory.json').read_text())
        state = json.loads((root / f'{mode}-container-state.json').read_text())
        bench = json.loads((root / f'{mode}-benchmark.json').read_text())
        samples = [json.loads(line) for line in (root / f'{mode}-telemetry.jsonl').read_text().splitlines()]
        assert not mem['aborted'] and not state['OOMKilled']
        assert len(mem['head_measurements']) == 1
        head = mem['head_measurements'][0]
        expected_rows = {'ko64k': 65536, 'ko128k': 131072, 'full': 248320}[mode]
        assert head['draft_head_shape'] == [expected_rows, 2560]
        assert head['shares_target_head'] == (mode == 'full')
        actual_kv = [int(n) for line in mem['cache_logs'] for n in re.findall(r'#tokens: (\d+)', line)]
        assert len(actual_kv) == 2 and set(actual_kv) == {run['kv_tokens']}
        assert any(f'max_mamba_cache_size: {run["mamba_slots"]},' in line for line in mem['cache_logs'])
        assert all(int(s.get('memory.events', {}).get('oom', 0)) == 0 for s in samples)
        memory_rows = [p for phase, p in mem['phases'].items() if phase != 'stopping']
        result['rows'].append({
            'mode': mode,
            'extra_head_mib': 0 if head['shares_target_head'] else head['draft_head_bytes'] / 2**20,
            'head_init_extra_peak_mib': (head['init_peak_allocated_bytes'] - head['before']['allocated_bytes']) / 2**20,
            'torch_after_head_gib': head['after_allocated_bytes'] / 2**30,
            'ready_idle_extra_gib': mem['phases']['ready_idle']['median_extra_memory_gib'],
            'post_idle_extra_gib': mem['phases']['post_idle']['median_extra_memory_gib'],
            'peak_extra_gib': max(p['max_extra_memory_gib'] for p in memory_rows),
            'min_available_gib': min(p['min_available_gib'] for p in memory_rows),
            'max_sample_gap_seconds': max(b['monotonic']-a['monotonic'] for a,b in zip(samples,samples[1:])),
            'max_swap_growth_mib': max(0, max(mem['baseline']['SwapFree']-s['SwapFree'] for s in samples)) / 2**20,
            'oom': False,
            'workload_checks': {x['name']: x['passed'] for x in bench['checks']},
            'api_tok_s': sum(r['usage']['completion_tokens'] for r in bench['rows']) / sum(r['elapsed_s'] for r in bench['rows']),
            'decode_tok_s_by_prompt': {r['name']: r['decode_tok_s'] for r in bench['rows']},
        })
    return result


if __name__ == '__main__':
    root = Path(sys.argv[1])
    result = summarize(root)
    (root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
