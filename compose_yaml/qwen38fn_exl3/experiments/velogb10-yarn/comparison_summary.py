"""Summarize measured A/B results without merging GPU allocations with host RSS."""
import argparse
import json
import os
import re
from pathlib import Path
import statistics


def memory(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    gpu = []
    for row in rows:
        sizes = []
        for line in row.get('gpu_processes', '').splitlines():
            try:
                sizes.append(int(line.split(',')[1].strip()))
            except (ValueError, IndexError):
                pass
        gpu.append(sum(sizes))
    return {
        'peak_GPU_GiB': max(gpu) / 1024,
        'min_system_available_GiB': min(r['host']['MemAvailable'] for r in rows) / 2**30,
        'max_process_swap_bytes': max((r.get('process', {}).get('VmSwap', 0) for r in rows), default=0) if any('process' in r for r in rows) else None,
        'max_container_swap_bytes': max((int(r.get('cgroup', {}).get('memory.swap.current', 0)) for r in rows), default=0) if any('cgroup' in r for r in rows) else None,
        'host_swap_write_MiB': (rows[-1]['swap_io_pages']['pswpout'] - rows[0]['swap_io_pages']['pswpout']) * os.sysconf('SC_PAGESIZE') / 2**20,
        'observed_seconds': rows[-1]['seconds'],
        'OOM_observed': any(r.get('oom', False) for r in rows),
    }


def engine(path):
    suite = json.loads((path / 'suite.json').read_text())
    byname = {r['name']: r for r in suite}
    def measure(prefix, field):
        values = [r[field] for r in suite if r['name'].startswith(prefix) and field in r]
        return {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'n': len(values)}
    groups = {
        'basic': ['arithmetic', 'json', 'units', 'japanese', 'chinese', 'logic'],
        'tool_calls_and_returns': [n for n in byname if n.startswith('tool_')],
        'literal_tool_documentation': [n for n in byname if n.startswith('quoted_xml_')],
        'vision': ['vision_quadrants', 'vision_followup'],
        'cold_prefill_and_prefix_cache': [n for n in byname if n.startswith(('cold_', 'cache_'))],
    }
    result = {
        'summary': json.loads((path / 'suite-summary.json').read_text()),
        'memory': memory(path / 'memory.jsonl'),
        'decode_tokens_per_second': {p: measure(p + '_', 'decode_tps') for p in ['korean', 'code']},
        'cold_TTFT_seconds': {p: measure('cold_' + p + '_', 'ttft') for p in ['1k', '8k', '32k']},
        'checks': {k: {'passed': sum(bool(byname[n].get('pass')) for n in names), 'total': len(names),
                       'failed': [n for n in names if not byname[n].get('pass')]} for k, names in groups.items()},
        'cache_16k': {n: {k: byname[n].get(k) for k in ['ttft', 'cached_tokens', 'text', 'pass', 'usage']} for n in ['cache_first', 'cache_repeat', 'cache_followup']},
    }
    for name in ['recall-1m', 'full-cache-repeat', 'full-cache-followup', 'post-full-reset', 'schema-probe']:
        file = path / (name + '.json')
        if file.exists():
            result[name] = json.loads(file.read_text())
    if (path / 'recall-32k.json').exists():
        result['recall-32k'] = json.loads((path / 'recall-32k.json').read_text())
    if (path / 'recall-320k.json').exists():
        result['recall-320k'] = json.loads((path / 'recall-320k.json').read_text())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    reference = a.root / 'exllama'
    candidate = a.root / 'velo'
    common = sorted(set(p.name for p in reference.glob('*.request.json')) & set(p.name for p in candidate.glob('*.request.json')))
    changed = [name for name in common if json.loads((reference / name).read_text()) != json.loads((candidate / name).read_text())]
    result = {
        'conditions': {'checkpoint': 'alesha-pro/Huihui-Qwen3.8-Flash-Next-abliterated-exl3-3bit-hq_h6_ng6',
                       'revision': '3b585c458f9fcf3322e76cff2c635c2cb81c5869', 'context': 1048576, 'yarn_factor': 4,
                       'KV': 'Q8', 'MTP_depth': 3, 'lanes': 1, 'CPU_affinity': '5-9,15-19', 'PLE': 'SSD',
                       'MTP_policy': {'exllama': 'Static depth 3, dynamic_draft false',
                                      'velo': 'Depth 3, DDS disabled; built-in profitability controller can disable MTP per request'},
                       'draft_vocabulary': {'exllama': 'Full 248320 tokens', 'velo': 'Leading 65536 tokens; not the Korean-aware ko64k shortlist'},
                       'vision': 'GPU resident', 'prefix_cache': 'on', 'checkpoint_cap_GiB': 4,
                       'cache_placement': {'exllama': 'CPU RAM', 'velo': 'GPU'},
                       'reference_image': 'sparktalk-qwen38fn_exl3:1.5.4-managed1',
                       'velo_commit': 'a6ad23d60e082ff9cca2bb78adda4ecee771388c plus local YaRN patch'},
        'exllama': engine(reference), 'velo': engine(candidate),
        'request_audit': {'common_requests': len(common), 'differing_requests': changed,
                          'note': 'Tool-return and vision/cache followup requests include each engine’s preceding output; speed and cold-prefill prompts must match exactly.'},
        'limits': ['Single GB10 and single existing checkpoint; other model services were stopped.',
                   'Three short speed samples per workload; 512-token caps, some outputs may stop earlier.',
                   'Both use the same full tokenizer; neither uses the ko64k draft-vocabulary shortlist.',
                   'MTP is configured at depth 3 in both, but Velo’s built-in profitability controller remains active.',
                   'TTFT and decode rates measured at the client including HTTP/SSE overhead.',
                   'Full-context quality is three-position synthetic recall, not a broad language benchmark.',
                   'GPU allocation and system MemAvailable are separate views of unified memory; never add them or add GPU allocation to process RSS.',
                   'No concurrent ASR/TTS/image/embedding residency qualification in this comparison.',
                   'Production profiles and original model weights remain unchanged.'],
        'artifacts': str(a.root),
    }
    if (a.root / 'velo-full-vocab/completed.json').exists():
        result['velo_full_vocabulary_control'] = engine(a.root / 'velo-full-vocab')
        result['limits'].append('The full-draft-vocabulary control re-ran short/320K checks with 1M preallocated KV and a saturated checkpoint cap, not the full 1M recall. Disabling the pruned draft head also disables its DHEAD screen/rescore path.')
        result['limits'].append('The 319990-token control reused 8192 tokens from its preceding 32K recall. Its raw total-prompt/TTFT ratio is not a fresh-prefill throughput comparison.')
        default_suite = {r['name']: r for r in json.loads((candidate / 'suite.json').read_text())}
        full_suite = {r['name']: r for r in json.loads((a.root / 'velo-full-vocab/suite.json').read_text())}
        speed_names = [n for n in default_suite if n.startswith(('korean_', 'code_'))]
        result['draft_vocabulary_control'] = {
            'only_launch_flag_changed': '--exl3-mtp-head-n 0',
            'same_visible_speed_outputs': {n: default_suite[n]['text'] == full_suite[n]['text'] for n in speed_names},
            'MTP_stats': {},
        }
        for label in ['velo', 'velo-full-vocab']:
            stats = re.findall(r'\[mtp-stats\].*', (a.root / label / 'server.log').read_text())
            result['draft_vocabulary_control']['MTP_stats'][label] = [
                {'case': name, 'drafted': int(re.search(r'drafted=(\d+)', line)[1]),
                 'accepted': int(re.search(r'accepted=(\d+)', line)[1]), 'raw': line}
                for name, line in zip(speed_names, stats[8:14])
            ]
    for name in ['mtp-comparison.json', 'vocab-coverage.json', 'exllama/logging-probe.json']:
        file = a.root / name
        if file.exists():
            result[name] = json.loads(file.read_text())
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: {'GPU': result[k]['memory']['peak_GPU_GiB'], 'KO': result[k]['decode_tokens_per_second']['korean']['median'],
                         'code': result[k]['decode_tokens_per_second']['code']['median'], '1M_pass': result[k].get('recall-1m', {}).get('pass')} for k in ['exllama', 'velo']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
