#!/usr/bin/env python3
"""Larger-budget completion follow-up; deliberately separate from primary results."""
import argparse
import json
from pathlib import Path

from summarize_broad_vocab import aggregate, coverage, expand
from vocab_memory import save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
    out=args.directory.resolve()
    selection=json.loads((out/'selection.json').read_text())
    tasks=json.loads((out/'tasks.json').read_text())['tasks']
    quality=json.loads((out/'quality-summary.json').read_text())
    maps=out.parent.parent/'2026-09-08-vocab-memory'
    sets={m:expand(maps/f'{m}.json') for m in ['ko64k','ko128k']}
    data={m:json.loads((out/f'{m}-benchmark.json').read_text()) for m in sets}
    assert data['ko64k']['tasks_sha256']==data['ko128k']['tasks_sha256']
    per_mode={}
    for mode,d in data.items():
        rows=d['rows']
        assert {r['id'] for r in rows}==set(selection['selected'])
        assert len(rows)==len(selection['selected'])
        per_mode[mode]=dict(**aggregate(rows),
            missing_answers=sum(r['thinking'] and '</think>' not in r['content'] for r in rows),
            coverage=coverage([i for r in rows for i in r['output_ids']],sets),
            quality=quality['modes'][mode])
    per_task=[]
    for task in tasks:
        pair={m:next(r for r in d['rows'] if r['id']==task['id']) for m,d in data.items()}
        assert pair['ko64k']['max_tokens']==pair['ko128k']['max_tokens']==task['max_tokens']
        grades={m:next(r for r in quality['rows'] if r['id']==task['id'] and r['mode']==m) for m in sets}
        per_task.append(dict(id=task['id'],domain=task['domain'],max_tokens=task['max_tokens'],
            decode_ratio_128_over_64=pair['ko128k']['decode_tok_s']/pair['ko64k']['decode_tok_s'],
            modes={m:dict(tokens=len(r['output_ids']),elapsed_s=r['elapsed_s'],decode_tok_s=r['decode_tok_s'],
                          truncated=r['truncated'],answer_present=not r['thinking'] or '</think>' in r['content'],
                          score=grades[m]['score'],notes=grades[m]['notes']) for m,r in pair.items()}))
    result=dict(selection=selection,modes=per_mode,per_task=per_task,
                limitation='One repeat, selected on primary 64K truncation; completion follow-up, not an unbiased domain speed sample.')
    save(out/'summary.json',result)
    lines=['# 출력 한도 확장 보충 검증','',
           '본 실험의 64K에서 출력이 잘리거나 최종 답변이 없었던 과제만 선정했습니다. '
           '동일한 입력으로 추론 8,192 / 일반 3,072토큰 한도를 적용해 설정별 한 번씩 실행했습니다. '
           '기존 96회 결과를 대체하지 않으며, 분야 전체의 평균 속도로 일반화하지 않습니다.','',
           '| 과제 | 출력 한도 | 64K 점수 | 128K 점수 | 64K 완료 시간 | 128K 완료 시간 | 잘림 64K/128K |',
           '|---|---:|---:|---:|---:|---:|---|']
    for t in per_task:
        a,b=t['modes']['ko64k'],t['modes']['ko128k']
        lines.append(f'| {t["id"]} | {t["max_tokens"]} | {a["score"]}/3 | {b["score"]}/3 | {a["elapsed_s"]:.1f}s | {b["elapsed_s"]:.1f}s | {a["truncated"]}/{b["truncated"]} |')
    lines+=['','점수는 사전에 고정한 세 기준의 충족 수이며 포괄적인 정답 인증이 아닙니다. '
           '코드의 기능 테스트 결과는 [code-grades.json](code-grades.json), 서술 판정은 '
           '[quality-summary.json](quality-summary.json)에 있습니다.','',
           '[선정 기준과 순서](selection.json) · [집계](summary.json) · '
           '[64K 응답](ko64k-benchmark.json) · [128K 응답](ko128k-benchmark.json)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(per_mode,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
