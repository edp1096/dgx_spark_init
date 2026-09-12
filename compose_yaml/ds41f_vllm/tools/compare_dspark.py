"""Compare proposal lengths without assuming equal generated token counts."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parents[1] / 'results'
sources = {
    5: ['io-heldout-1.json', 'io-heldout-2.json'],
    3: ['dspark3-heldout-0.json', 'dspark3-heldout-1.json'],
    2: ['dspark2-heldout-0.json', 'dspark2-heldout-1.json'],
}
runs = {n: [json.loads((root / name).read_text()) for name in names]
        for n, names in sources.items()}
baseline = runs[5][0]
report = {
    'settings': 'TP2, uniform224, direct batch_overlap, expert graphs, eager model',
    'method': 'One excluded warmup and two timed suites per length; each suite starts '
              'with empty expert slots; each of four prompts is repeated immediately. '
              'Fixed order 5 then 3 then 2, not an interleaved length experiment.',
    'warning': 'Generated wording/token counts can differ. Total time ratios alone '
               'are not controlled same-output speedups. Small serial test only.',
    'lengths': {},
}
names = ['merge_intervals', 'dns', 'korean_cache_database', 'monthly_sql']
for n, trials in runs.items():
    assert all(len(t) == len(baseline) for t in trials)
    rows = []
    for i, reference in enumerate(baseline):
        values = [t[i] for t in trials]
        assert all(v['prompt'] == reference['prompt'] for v in values)
        rows.append({
            'workload': names[i // 2], 'immediate_repeat': bool(i % 2),
            'completion_tokens': [v['usage']['completion_tokens'] for v in values],
            'same_text_as_length5': all(v['text'] == reference['text'] for v in values),
            'stable_between_trials': values[0]['text'] == values[1]['text'],
            'finish_reasons': [v['finish_reason'] for v in values],
            'mean_total_seconds': mean(v['total_seconds'] for v in values),
            'mean_ttft_seconds': mean(v['ttft_seconds'] for v in values),
            'mean_decode_tps': mean(v['decode_tokens_per_second'] for v in values),
        })
    totals = [sum(v['total_seconds'] for v in t) for t in trials]
    tokens = [sum(v['usage']['completion_tokens'] for v in t) for t in trials]
    groups = {}
    for label, parity in [('first_requests', 0), ('immediate_repeats', 1)]:
        group = [v for t in trials for i, v in enumerate(t) if i % 2 == parity]
        groups[label] = {
            'aggregate_overall_tps': sum(v['usage']['completion_tokens'] for v in group)
                                     / sum(v['total_seconds'] for v in group),
            'aggregate_decode_tps': sum(v['usage']['completion_tokens']
                                       - v['first_content_event_tokens'] for v in group)
                                    / sum(v['total_seconds'] - v['ttft_seconds'] for v in group),
        }
    report['lengths'][str(n)] = {
        'sources': sources[n], 'timed_total_seconds': totals,
        'mean_total_seconds': mean(totals), 'completion_tokens_per_suite': tokens,
        'aggregate_overall_tps': sum(tokens) / sum(totals),
        'groups': groups, 'rows': rows,
    }
path = root / 'dspark-length-comparison.json'
path.write_text(json.dumps(report, indent=2) + '\n')
for n, result in report['lengths'].items():
    print(n, json.dumps({k: v for k, v in result.items() if k != 'rows'}))
    for row in result['rows']:
        print(' ', row['workload'], row['immediate_repeat'], row['completion_tokens'],
              round(row['mean_decode_tps'], 3), row['same_text_as_length5'])
