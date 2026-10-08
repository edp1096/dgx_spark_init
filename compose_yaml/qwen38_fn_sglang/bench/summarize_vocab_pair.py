#!/usr/bin/env python3
"""Compare complete paired workloads with explicit vocabulary mode names."""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import statistics

from summarize_broad_vocab import aggregate, expand, geomean
from grade_broad_code import answer_text
from vocab_memory import save


def read(path):
    return json.loads(path.read_text())


def answer_ids(row, close_think):
    ids = row['output_ids']
    if not row['thinking']:
        return ids
    return ids[ids.index(close_think) + 1:] if close_think in ids else []


def has_answer(row):
    return (not row['thinking'] or '</think>' in row['content']) and bool(answer_text(row['content']))


def quality(rows):
    return dict(requests=len(rows), points=sum(r['score'] for r in rows),
                max_points=3 * len(rows), full_pass=sum(r['score'] == 3 for r in rows),
                material_errors=sum(r['material_error'] for r in rows))


def thermal(directory, rows):
    intervals = [(datetime.datetime.fromisoformat(r['started']),
                  datetime.datetime.fromisoformat(r['finished'])) for r in rows]
    observations, invalid = {}, 0
    paths = [directory / 'thermal.jsonl', *sorted((directory / 'attempts').glob('*/thermal.jsonl'))]
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                timestamp = datetime.datetime.fromisoformat(row['time'])
            except (ValueError, KeyError):
                invalid += 1
                continue
            if any(a <= timestamp <= b for a, b in intervals):
                observations[row['time']] = row
    metrics = {}
    for key in ['temperature_c', 'gpu_util_pct', 'power_w', 'sm_clock_mhz']:
        values = []
        for row in observations.values():
            try:
                values.append(float(row['values'][key]))
            except (ValueError, KeyError, TypeError):
                pass
        if values:
            metrics[key] = dict(samples=len(values), min=min(values), max=max(values),
                                median=statistics.median(values))
    return dict(metrics=metrics, invalid_lines_excluded=invalid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--reference-mode', default='ko64k')
    ap.add_argument('--candidate-mode', default='ko32k')
    args = ap.parse_args()
    out, ref = args.directory.resolve(), args.reference.resolve()
    rm, cm = args.reference_mode, args.candidate_mode
    modes = [rm, cm]
    assert (out / 'complete.json').exists(), 'Candidate trial is not complete'
    assert read(out / 'run.json')['image'] == read(ref / 'run.json')['image']
    assert hashlib.sha256((out / '.runtime/eagle_worker_v2.py').read_bytes()).digest() == hashlib.sha256((ref / '.runtime/eagle_worker_v2.py').read_bytes()).digest()
    tasks = read(out / 'tasks.json')['tasks']
    taskhash = hashlib.sha256((out / 'tasks.json').read_bytes()).hexdigest()
    assert hashlib.sha256((ref / 'tasks.json').read_bytes()).hexdigest() == taskhash
    assert hashlib.sha256((out / 'prepared.json').read_bytes()).hexdigest() == hashlib.sha256((ref / 'prepared.json').read_bytes()).hexdigest()
    expected = {(t['id'], repeat) for t in tasks for repeat in range(2)}
    allrows, infos = {}, {}
    for mode, directory in [(rm, ref), (cm, out)]:
        data = read(directory / f'{mode}-benchmark.json')
        assert data['tasks_sha256'] == taskhash and data['rounds'] == 2
        rows = data['rows']
        assert len(rows) == len(expected) and {(r['id'], r['repeat']) for r in rows} == expected
        for row in rows:
            assert len(row['output_ids']) == row['meta_info']['completion_tokens']
        allrows[mode] = rows
        infos[mode] = read(directory / f'{mode}-server-info.json')
    for key in ['model_path', 'tokenizer_path', 'context_length', 'max_total_tokens',
                'max_mamba_cache_size', 'speculative_algorithm', 'speculative_num_steps',
                'speculative_num_draft_tokens', 'speculative_eagle_topk', 'mem_fraction_static']:
        assert infos[rm].get(key) == infos[cm].get(key), key
    assert infos[cm]['context_length'] == infos[cm]['max_total_tokens'] == 65536
    assert infos[cm]['max_mamba_cache_size'] == 18
    close_think = read(out / 'token-metadata.json')['special_token_ids']['</think>']
    source_maps = ref.parent / '2026-09-08-vocab-memory'
    sets = {rm: expand(source_maps / f'{rm}.json') if (source_maps / f'{rm}.json').exists() else set(read(ref / '.runtime' / f'{rm}-ids.json')),
            cm: set(read(out / '.runtime' / f'{cm}-ids.json'))}
    assert len(sets[cm]) == int(cm.removeprefix('ko').removesuffix('k')) * 1024 and sets[cm] <= sets[rm]
    grades = read(out / 'quality-summary.json')['rows']
    assert len(grades) == 2 * len(expected)
    grade_by_key = {(r['mode'], r['id'], r['repeat']): r for r in grades}
    assert len(grade_by_key) == len(grades)
    by_key = {m: {(r['id'], r['repeat']): r for r in rows} for m, rows in allrows.items()}
    # Generalized coverage uses the same counts for both sets; the older coverage
    # helper's recovered_by_128k field is intentionally not reused for 32K.
    def scope_coverage(ids):
        return dict(tokens=len(ids), outside={m: sum(i not in s for i in ids) for m, s in sets.items()},
                    outside_pct={m: 100 * sum(i not in s for i in ids) / len(ids) if ids else 0
                                 for m, s in sets.items()})
    mode_stats = {}
    for mode, rows in allrows.items():
        mode_stats[mode] = dict(**aggregate(rows),
            missing_final=sum(not has_answer(r) for r in rows),
            quality=quality([g for g in grades if g['mode'] == mode]),
            coverage=scope_coverage([i for r in rows for i in r['output_ids']]),
            final_coverage=scope_coverage([i for r in rows for i in answer_ids(r, close_think)]),
            thermal=thermal(ref if mode == rm else out, rows))
    pairs = []
    for ident, repeat in sorted(expected):
        a, b = [by_key[m][ident, repeat] for m in modes]
        assert a['max_tokens'] == b['max_tokens'] and a['thinking'] == b['thinking']
        assert a['input_tokens'] == b['input_tokens']
        pairs.append(dict(id=ident, repeat=repeat, domain=a['domain'],
            ratio_candidate_over_reference=b['decode_tok_s'] / a['decode_tok_s'],
            elapsed_ratio_candidate_over_reference=b['elapsed_s'] / a['elapsed_s'],
            both_finished=all(not r['truncated'] and has_answer(r) for r in [a, b]),
            exact_output_match=a['output_ids'] == b['output_ids'],
            grades={m: grade_by_key[m, ident, repeat] for m in modes}))
    finished = [p['ratio_candidate_over_reference'] for p in pairs if p['both_finished']]
    finished_by_task = collections.defaultdict(list)
    for pair in pairs:
        if pair['both_finished']:
            finished_by_task[pair['id']].append(pair['ratio_candidate_over_reference'])
    domains = {}
    for domain in sorted({t['domain'] for t in tasks}):
        ps = [p for p in pairs if p['domain'] == domain]
        domains[domain] = dict(pairs=len(ps),
            ratio_candidate_over_reference=geomean([p['ratio_candidate_over_reference'] for p in ps]),
            modes={m: dict(**aggregate([r for r in rows if r['domain'] == domain]),
                quality=quality([g for g in grades if g['mode'] == m and g['domain'] == domain]))
                for m, rows in allrows.items()})
    memory = read(out / f'{cm}-memory.json')
    assert not memory['aborted'] and not memory['monitor_error']
    assert not read(out / f'{cm}-container-state.json')['OOMKilled']
    telemetry = [json.loads(line) for line in (out / f'{cm}-telemetry.jsonl').read_text().splitlines()]
    assert not any(int(r.get('memory.events', {}).get('oom_kill', 0)) for r in telemetry)
    memory['telemetry_summary'] = dict(
        min_available_gib=min(r['MemAvailable'] for r in telemetry) / 2**30,
        max_swap_growth_mib=max(0, memory['baseline']['SwapFree'] - min(r['SwapFree'] for r in telemetry)) / 2**20)
    result = dict(reference_mode=rm, candidate_mode=cm, tasks_sha256=taskhash, modes=mode_stats,
        pairs=pairs, domains=domains,
        pair_geomean_ratio_candidate_over_reference=geomean([p['ratio_candidate_over_reference'] for p in pairs]),
        complete_pairs=len(finished), complete_pair_geomean_ratio=geomean(finished) if finished else None,
        complete_task_balanced_geomean_ratio=geomean([geomean(v) for v in finished_by_task.values()]) if finished_by_task else None,
        quality_pair_comparison=dict(
            candidate_higher=sum(p['grades'][cm]['score'] > p['grades'][rm]['score'] for p in pairs),
            reference_higher=sum(p['grades'][cm]['score'] < p['grades'][rm]['score'] for p in pairs),
            tied=sum(p['grades'][cm]['score'] == p['grades'][rm]['score'] for p in pairs)),
        exact_matches=sum(p['exact_output_match'] for p in pairs), candidate_memory=memory,
        limitations=['Reference measured before host reboot, candidate after it; sessions/order/thermal conditions differ.',
                     'Reference includes a process restart; prompts/settings match but this is not an interleaved trial.',
                     '48 paired requests are two repeats of 24 tasks, not 48 independent task types.',
                     'Three binary checks per task; prose is assistant judgment, not independent expert validation.',
                     'Token-limit and missing-final failures are included; no supplementary longer-budget trial is pooled.',
                     'Different output token sequences and lengths can influence speed; no causal vocabulary-only claim.',
                     'Speculative shortlist changes draft candidates, not the target model vocabulary or context length.'])
    save(out / 'comparison-summary.json', result)
    lines = [f'# {cm} 실제 실행과 기존 {rm} 비교', '',
        f'동일한 24개 문항을 두 번씩 실행한 48쌍이다. {cm}는 새 실행, {rm}는 저장된 본 실험이다. 후보 어휘 외 생성 설정과 입력 토큰은 일치 여부를 검사했다.', '',
        f'| 항목 | {rm} | {cm} |', '|---|---:|---:|']
    interpretation = out / 'interpretation.md'
    if interpretation.exists():
        lines[4:4] = [interpretation.read_text().strip(), '']
    for label, fn in [
        ('루브릭 점수', lambda s: f"{s['quality']['points']}/{s['quality']['max_points']}"),
        ('3개 기준 모두 통과', lambda s: str(s['quality']['full_pass'])),
        ('서술형 중요 오류 표시', lambda s: str(s['quality']['material_errors'])),
        ('출력 한도 도달', lambda s: str(s['truncated'])),
        ('최종 답변 없음', lambda s: str(s['missing_final'])),
        ('생성 토큰 수', lambda s: str(s['tokens'])),
        ('응답 대기시간 합계(분)', lambda s: f"{s['elapsed_s']/60:.2f}"),
        ('합산 decode tok/s', lambda s: f"{s['pooled_decode_tok_s']:.2f}"),
        ('합산 전체 응답 tok/s', lambda s: f"{s['pooled_end_to_end_tok_s']:.2f}"),
        ('첫 토큰 지연 중앙값(초)', lambda s: f"{s['median_ttft_s']:.3f}"),
        ('draft 수락률', lambda s: f"{100*s['spec_accept_rate']:.2f}%"),
        ('검증당 출력 토큰', lambda s: f"{s['spec_accept_length']:.3f}")]:
        lines.append('| ' + label + ' | ' + ' | '.join(fn(mode_stats[m]) for m in modes) + ' |')
    ratio = result['pair_geomean_ratio_candidate_over_reference']
    lines += ['', f'문항·반복별 생성 속도비({cm}/{rm})의 기하평균은 **{ratio:.4f}**이다. 1보다 크면 {cm}가 빠르다.',
        '중요 오류 표시는 서술형 검토의 별도 지표다. 코드 미제출과 집계 오답은 루브릭 점수에 반영되며 이 표시 수에는 자동으로 포함되지 않는다.',
        f'양쪽 모두 한도에 걸리지 않고 답변을 낸 {len(finished)}쌍의 속도비는 {geomean(finished):.4f}이다.' if finished else '양쪽 완료 조건을 충족하는 쌍이 없다.',
        f"완전히 동일한 출력 토큰열은 {result['exact_matches']}/48쌍이다.", '',
        '## 분야별 비교', '', f'| 분야 | 쌍 수 | {rm} 점수 | {cm} 점수 | 속도비 {cm}/{rm} |', '|---|---:|---:|---:|---:|']
    for domain, d in domains.items():
        a, b = [d['modes'][m]['quality'] for m in modes]
        lines.append(f"| {domain} | {d['pairs']} | {a['points']}/{a['max_points']} | {b['points']}/{b['max_points']} | {d['ratio_candidate_over_reference']:.3f} |")
    lines += ['', '## 문항별 비교', '', f'| 문항 | 반복 | {rm} 점수/3 | {cm} 점수/3 | 속도비 |', '|---|---:|---:|---:|---:|']
    for p in pairs:
        lines.append(f"| {p['id']} | {p['repeat']+1} | {p['grades'][rm]['score']} | {p['grades'][cm]['score']} | {p['ratio_candidate_over_reference']:.3f} |")
    lines += ['', '## 측정 범위와 한계', ''] + ['- ' + s for s in result['limitations']]
    lines += ['', '## 응답 생성 중 온도와 클럭', '',
              '| 모드 | GPU 온도 중앙값/최대(°C) | SM 클럭 중앙값(MHz) |', '|---|---:|---:|']
    for mode in modes:
        metrics = mode_stats[mode]['thermal']['metrics']
        temp, clock = metrics.get('temperature_c'), metrics.get('sm_clock_mhz')
        if temp and clock:
            lines.append(f"| {mode} | {temp['median']:.1f}/{temp['max']:.1f} | {clock['median']:.0f} |")
    lines += ['', '10초 간격 관측이다. 손상된 이전 온도 로그 행은 제외했다. 클럭·온도 차이가 있다면 속도 차이의 교란 요인으로 보아야 한다.']
    ms = memory['telemetry_summary']
    lines += ['', f'## {cm} 메모리 관측', '',
              f"로딩을 포함한 최소 가용 메모리 {ms['min_available_gib']:.2f}GiB, 시작 대비 최대 추가 스왑 {ms['max_swap_growth_mib']:.2f}MiB. 감시 중단과 컨테이너 OOM은 없었다.",
              f'후보 출력층 가중치는 {len(sets[cm])*2560*2/2**20:.0f}MiB로, 같은 BF16 {rm} 출력층의 {len(sets[rm])*2560*2/2**20:.0f}MiB보다 작다. 이는 모델 전체 메모리가 같은 비율로 감소한다는 뜻이 아니다.']
    lines += ['', '## 자료', '', '- [개별 점수와 판정 이유](quality-summary.json)',
        '- [코드 제출·기능 검사 결과](code-grades.json)',
        '- [문서 집계 답안과 정답 대조](ledger-grades.json) / [정답 독립 검산](ledger-reference-audit.json)',
        '- [전체 집계·토큰 범위·메모리 측정](comparison-summary.json)',
        f'- [{cm} 원문]({cm}-benchmark.json) / [{rm} 원문]({rm}-benchmark.json)',
        '- [실행 계획](PLAN.md)', '']
    (out / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(modes=mode_stats, ratio=ratio, complete_pairs=len(finished)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
