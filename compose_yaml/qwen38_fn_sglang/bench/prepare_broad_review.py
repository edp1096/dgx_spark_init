#!/usr/bin/env python3
"""Make mode-hidden, deduplicated prose-review packets and objective ledger grades."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from grade_broad_code import answer_text
from vocab_memory import save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
    ap.add_argument('--modes', nargs='+', default=['ko64k','ko128k'])
    ap.add_argument('--partial',action='store_true',help='Review completed rows while the second mode is still running')
    args=ap.parse_args()
    out=args.directory.resolve()
    tasks={t['id']:t for t in json.loads((out/'tasks.json').read_text())['tasks']}
    unique={};mapping=[];ledger=[]
    mode_rows={}
    for mode in args.modes:
        path=out/f'{mode}-benchmark.json'
        if path.exists():
            rows=json.loads(path.read_text())['rows']
        elif args.partial:
            text=(out/f'{mode}-benchmark.jsonl').read_text()
            lines=text.splitlines()
            if not text.endswith('\n'):
                lines=lines[:-1]
            rows=[json.loads(line) for line in lines]
        else:
            raise RuntimeError('Both completed mode files are required')
        mode_rows[mode]=rows
    eligible=set.intersection(*({r['id'] for r in rows} for rows in mode_rows.values()))
    for mode, rows in mode_rows.items():
        for row in rows:
            if args.partial and row['id'] not in eligible:
                continue
            task=tasks[row['id']]
            answer=answer_text(row['content'])
            if row['thinking'] and '</think>' not in row['content']:
                answer=''
            key=hashlib.sha256((row['id']+'\0'+answer).encode()).hexdigest()
            mapping.append(dict(mode=mode,id=row['id'],repeat=row['repeat'],answer_sha256=key,
                                truncated=row['truncated']))
            if row['id'] in ['code_shortlist','code_telemetry']:
                continue
            if row['id']=='long_ledger':
                obj=None
                for offset,c in enumerate(answer):
                    if c=='{':
                        try:obj=json.JSONDecoder().raw_decode(answer[offset:])[0];break
                        except json.JSONDecodeError:pass
                groups=[isinstance(obj,dict) and all(isinstance(obj.get(region),dict) and
                    obj[region].get(field)==expected[field] for region,expected in task['expected'].items())
                    for field in ['orders','units','revenue']]
                ledger.append(dict(mode=mode,id=row['id'],repeat=row['repeat'],groups=groups,
                                   parsed=obj,expected=task['expected']))
                continue
            unique[key]=dict(answer_sha256=key,task_id=row['id'],rubric=task['rubric'],
                             sources=task['sources'],answer=answer)
    packets=list(unique.values());random.Random(80216).shuffle(packets)
    for p in packets:p['review_id']='V'+p['answer_sha256'][:16]
    assert len({p['review_id'] for p in packets})==len(packets)
    save(out/'review-packets.json',packets)
    save(out/'review-mapping.json',mapping)
    save(out/'ledger-grades.json',ledger)
    print(f'{len(packets)} distinct prose answers to review; {len(ledger)} ledger grades saved')


if __name__=='__main__':main()
