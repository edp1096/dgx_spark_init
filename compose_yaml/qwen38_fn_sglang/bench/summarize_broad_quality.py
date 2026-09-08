#!/usr/bin/env python3
"""Combine frozen-rubric prose reviews and independently executed objective checks."""
import argparse
import collections
import json
from pathlib import Path

from vocab_memory import save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
    ap.add_argument('--modes', nargs='+', default=['ko64k','ko128k'])
    args=ap.parse_args()
    out=args.directory.resolve()
    tasks={t['id']:t for t in json.loads((out/'tasks.json').read_text())['tasks']}
    packets={p['review_id']:p for p in json.loads((out/'review-packets.json').read_text())}
    reviews=json.loads((out/'prose-reviews.json').read_text())
    assert len(reviews)==len(packets)
    assert {r['review_id'] for r in reviews}==set(packets)
    byhash={packets[r['review_id']]['answer_sha256']:r for r in reviews}
    objective={}
    for filename in ['code-grades.json','ledger-grades.json']:
        for r in json.loads((out/filename).read_text()):
            objective[r['mode'],r['id'],r['repeat']]=r
    rows=[]
    for r in json.loads((out/'review-mapping.json').read_text()):
        key=r['mode'],r['id'],r['repeat']
        grade=objective.get(key) or byhash[r['answer_sha256']]
        groups=grade['groups']
        assert len(groups)==3 and all(isinstance(v,bool) for v in groups)
        rows.append(dict(**r,domain=tasks[r['id']]['domain'],groups=groups,score=sum(groups),
                         notes=grade.get('notes',grade.get('error','')),
                         material_error=grade.get('material_error',False),
                         method=('completion check: no submitted code' if grade.get('execution_attempted') is False
                                 else 'executed checks' if key in objective else 'assistant rubric review')))
    result={}
    for mode in args.modes:
        rs=[r for r in rows if r['mode']==mode]
        result[mode]=dict(points=sum(r['score'] for r in rs),max_points=len(rs)*3,
                          full_pass_requests=sum(r['score']==3 for r in rs),requests=len(rs),
                          material_errors=sum(r['material_error'] for r in rs),
                          domains={d:dict(points=sum(r['score'] for r in rs if r['domain']==d),
                              max_points=3*sum(r['domain']==d for r in rs)) for d in sorted({r['domain'] for r in rs})})
    save(out/'quality-summary.json',dict(modes=result,rows=rows,
        limitations=['Three binary rubric checks per task are a narrow correctness screen, not comprehensive domain certification.',
                     'Prose judgments by the assistant with mode labels hidden, not independent expert reviewers.',
                     'Identical final answers deduplicated for consistent scoring; no extra weight beyond their observed requests.',
                     'Truncation reported separately; partial final answers only receive criteria they actually satisfy.']))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
