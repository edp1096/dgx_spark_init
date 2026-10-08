#!/usr/bin/env python3
"""Three-way paired summary for the preserved 64K, 32K and new 16K trials."""
import argparse
import hashlib
import json
from pathlib import Path
from summarize_broad_vocab import aggregate, expand, geomean
from summarize_vocab_pair import answer_ids, has_answer, quality, thermal
from vocab_memory import save


def read(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    out = args.directory.resolve()
    dirs = dict(ko16k=out, ko32k=out.parent/'2026-09-09-ko32k-vs64k',
                ko64k=out.parent/'2026-09-08-broad-vocab')
    modes = list(dirs)
    assert read(out/'complete.json')['production_container_unchanged']
    tasks = read(out/'tasks.json')['tasks']
    expected = {(t['id'], repeat) for t in tasks for repeat in range(2)}
    close = read(out/'token-metadata.json')['special_token_ids']['</think>']
    current_grades = read(out/'quality-summary.json')['rows']
    prior_grades = read(dirs['ko32k']/'quality-summary.json')['rows']
    qs = {m: [r for r in (prior_grades if m == 'ko32k' else current_grades) if r['mode'] == m] for m in modes}
    old64 = {(r['id'], r['repeat']): r for r in prior_grades if r['mode'] == 'ko64k'}
    for r in qs['ko64k']:
        old = old64[r['id'], r['repeat']]
        assert r['groups'] == old['groups'] and r['material_error'] == old['material_error']
    rows, infos = {}, {}
    for m, d in dirs.items():
        for filename in ['tasks.json', 'prepared.json']:
            assert hashlib.sha256((d/filename).read_bytes()).digest() == hashlib.sha256((out/filename).read_bytes()).digest()
        assert read(d/'run.json')['image'] == read(out/'run.json')['image']
        data = read(d/f'{m}-benchmark.json')
        assert data['rounds'] == 2
        rows[m] = data['rows']
        assert len(rows[m]) == len(qs[m]) == 48
        assert {(r['id'], r['repeat']) for r in rows[m]} == expected
        assert {(r['id'], r['repeat']) for r in qs[m]} == expected
        assert all(len(r['output_ids']) == r['meta_info']['completion_tokens'] for r in rows[m])
        infos[m] = read(d/f'{m}-server-info.json')
    fields = ['model_path', 'tokenizer_path', 'context_length', 'max_total_tokens', 'max_mamba_cache_size',
              'speculative_algorithm', 'speculative_num_steps', 'speculative_num_draft_tokens',
              'speculative_eagle_topk', 'mem_fraction_static']
    for field in fields:
        assert all(infos[m].get(field) == infos['ko64k'].get(field) for m in modes), field
    sets = {m: set(read(d/'.runtime'/f'{m}-ids.json')) for m,d in dirs.items() if m != 'ko64k'}
    sets['ko64k'] = expand(out.parent/'2026-09-08-vocab-memory/ko64k.json')
    assert sets['ko16k'] <= sets['ko32k'] <= sets['ko64k']
    assert [len(sets[m]) for m in modes] == [16384,32768,65536]
    def coverage(ids):
        return dict(tokens=len(ids), outside_pct={m: 100*sum(i not in s for i in ids)/len(ids) if ids else 0 for m,s in sets.items()})
    stats = {}
    for m, rs in rows.items():
        stats[m] = dict(**aggregate(rs), quality=quality(qs[m]), missing_final=sum(not has_answer(r) for r in rs),
            head_mib=len(sets[m])*2560*2/2**20, thermal=thermal(dirs[m],rs),
            coverage=coverage([i for r in rs for i in r['output_ids']]),
            final_coverage=coverage([i for r in rs for i in answer_ids(r,close)]))
    save(out/'three-way-quality.json',dict(modes={m:quality(qs[m]) for m in modes},rows=[r for m in modes for r in qs[m]]))
    lookup = {m: {(r['id'],r['repeat']):r for r in rs} for m,rs in rows.items()}
    qlookup = {m: {(r['id'],r['repeat']):r for r in rs} for m,rs in qs.items()}
    pairwise = {}
    for candidate, reference in [('ko16k','ko64k'),('ko16k','ko32k'),('ko32k','ko64k')]:
        ratios, finished, deltas = [], [], []
        for key in sorted(expected):
            a,b = lookup[reference][key], lookup[candidate][key]
            assert a['input_tokens']==b['input_tokens'] and a['max_tokens']==b['max_tokens'] and a['thinking']==b['thinking']
            ratio = b['decode_tok_s']/a['decode_tok_s'];ratios.append(ratio)
            if all(not r['truncated'] and has_answer(r) for r in [a,b]):finished.append(ratio)
            deltas.append(qlookup[candidate][key]['score']-qlookup[reference][key]['score'])
        pairwise[f'{candidate}_over_{reference}'] = dict(geomean_decode_ratio=geomean(ratios),
            complete_pairs=len(finished), complete_geomean_ratio=geomean(finished) if finished else None,
            quality_higher=sum(d>0 for d in deltas), quality_lower=sum(d<0 for d in deltas), quality_tied=sum(d==0 for d in deltas))
    domains = {}
    for domain in sorted({t['domain'] for t in tasks}):
        keys = [key for key in sorted(expected) if lookup['ko64k'][key]['domain']==domain]
        domains[domain] = dict(pairs=len(keys), modes={m:dict(**aggregate([lookup[m][k] for k in keys]),
            quality=quality([qlookup[m][k] for k in keys])) for m in modes},
            ratio16_over64=geomean([lookup['ko16k'][k]['decode_tok_s']/lookup['ko64k'][k]['decode_tok_s'] for k in keys]),
            ratio16_over32=geomean([lookup['ko16k'][k]['decode_tok_s']/lookup['ko32k'][k]['decode_tok_s'] for k in keys]))
    contexts = {m: read(d/'system-context.json') if (d/'system-context.json').exists() else {'run_started':read(d/'run.json')['started']} for m,d in dirs.items()}
    result = dict(modes=stats,pairwise=pairwise,domains=domains,task_count=24,repeats=2,system_contexts=contexts,
        limitations=['Sequential runs across boot sessions; another host reboot occurred after the 32K trial. Clock/thermal/order/output-length effects remain.',
                     'An earlier 16K load ended with host resets and zero responses. These metrics describe the explicitly requested retry only, not proof of hardware stability.',
                     '48 requests are two repeats of 24 task types, not 48 independent scenarios.',
                     'Prose uses frozen assistant-rated rubrics, not independent expert judgments.',
                     'Only draft vocabulary changes; target vocabulary/context is not reduced.',
                     'Token-limit and missing-final failures retained. No selected extended-budget requests are pooled.'])
    save(out/'three-way-summary.json',result)
    lines=['# ko16k · ko32k · ko64k 실제 비교','',
        '동일한 24개 문항을 두 번씩 실행한 48쌍이다. 16k만 새로 실행하고 32k·64k의 저장 결과를 대조했다. 입력 토큰·생성 설정·이미지와 기존 64k 채점 유지 여부를 검증했다.','']
    interpretation=out/'three-way-interpretation.md'
    if interpretation.exists():lines += [interpretation.read_text().strip(),'']
    lines += ['| 항목 | ko16k | ko32k | ko64k |','|---|---:|---:|---:|']
    for label,fn in [
        ('고정 기준 점수',lambda s:f"{s['quality']['points']}/144"),
        ('3개 기준 모두 통과',lambda s:f"{s['quality']['full_pass']}/48"),
        ('서술형 중요 오류 표시',lambda s:str(s['quality']['material_errors'])),
        ('최종 답변 미제출',lambda s:str(s['missing_final'])),
        ('출력 한도 도달',lambda s:str(s['truncated'])),
        ('합산 decode tok/s',lambda s:f"{s['pooled_decode_tok_s']:.2f}"),
        ('합산 전체 응답 tok/s',lambda s:f"{s['pooled_end_to_end_tok_s']:.2f}"),
        ('문항 속도의 산술평균 tok/s',lambda s:f"{s['mean_request_decode_tok_s']:.2f}"),
        ('첫 토큰 지연 중앙값(초)',lambda s:f"{s['median_ttft_s']:.3f}"),
        ('초안 수락률',lambda s:f"{100*s['spec_accept_rate']:.2f}%"),
        ('생성 토큰 수',lambda s:str(s['tokens'])),
        ('응답 대기시간 합계(분)',lambda s:f"{s['elapsed_s']/60:.2f}"),
        ('후보 출력층 가중치(MiB)',lambda s:f"{s['head_mib']:.0f}"),
        ('관측 SM 클럭 중앙값(MHz)',lambda s:f"{s['thermal']['metrics']['sm_clock_mhz']['median']:.0f}"),
        ('관측 GPU 온도 중앙값/최대(°C)',lambda s:f"{s['thermal']['metrics']['temperature_c']['median']:.0f}/{s['thermal']['metrics']['temperature_c']['max']:.0f}")]:
        lines.append('| '+label+' | '+' | '.join(fn(stats[m]) for m in modes)+' |')
    lines += ['', '합산 처리량과 문항별 평균은 가중 방식이 다르다. 중요 오류 표시는 수동 서술형 검토의 별도 지표이며 코드/집계 실패는 점수로 반영한다.','','## 같은 요청끼리 속도·점수 비교','','| 비교 | 속도비 기하평균 | 양쪽 완료 쌍 수 | 완료 쌍 속도비 | 후보 점수 우세/열세/동점 |','|---|---:|---:|---:|---:|']
    for name,r in pairwise.items():
        finished=f"{r['complete_geomean_ratio']:.4f}" if r['complete_geomean_ratio'] else 'N/A'
        lines.append(f"| {name} | {r['geomean_decode_ratio']:.4f} | {r['complete_pairs']} | {finished} | {r['quality_higher']}/{r['quality_lower']}/{r['quality_tied']} |")
    lines += ['','## 분야별 비교','','| 분야 | 쌍 수 | 16k 점수 | 32k 점수 | 64k 점수 | 속도비16/32 | 속도비16/64 |','|---|---:|---:|---:|---:|---:|---:|']
    for domain,d in domains.items():
        scores=' | '.join(f"{d['modes'][m]['quality']['points']}/{d['modes'][m]['quality']['max_points']}" for m in modes)
        lines.append(f"| {domain} | {d['pairs']} | {scores} | {d['ratio16_over32']:.3f} | {d['ratio16_over64']:.3f} |")
    lines += ['','## 한계','']+['- '+s for s in result['limitations']]
    lines += ['','## 상세 자료','', '- [3개 설정 집계 JSON](three-way-summary.json)',
        '- [3개 설정의 144개 응답 점수·판정 근거](three-way-quality.json)',
        '- [16k·64k 개별 점수와 근거](quality-summary.json)',
        '- [32k 개별 점수와 근거](../2026-09-09-ko32k-vs64k/quality-summary.json)',
        '- [16k 원문](ko16k-benchmark.json) / [32k 원문](../2026-09-09-ko32k-vs64k/ko32k-benchmark.json) / [64k 원문](ko64k-benchmark.json)',
        '- [16k 메모리 관측](ko16k-memory.json) / [코드 검사](code-grades.json) / [원장 대조](ledger-grades.json)',
        '- [실행 계획](PLAN.md)','']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(pairwise=pairwise,modes={m:dict(quality=s['quality'],decode=s['pooled_decode_tok_s'],accept=s['spec_accept_rate']) for m,s in stats.items()}),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
