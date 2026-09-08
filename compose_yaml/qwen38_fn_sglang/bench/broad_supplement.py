"""Paired larger-output-budget follow-up, selected before observing any 128K output."""
import copy
import hashlib
import json
import os
import shutil

from compare import flush
from vocab_memory import save


def prepare(parent):
    baseline=json.loads((parent/'ko64k-benchmark.json').read_text())['rows']
    assert len(baseline)==48
    selected={r['id'] for r in baseline if r['truncated'] or (r['thinking'] and '</think>' not in r['content'])}
    if not selected:
        return None
    out=parent/'extended'
    if (out/'prepared.json').exists():
        selection=json.loads((out/'selection.json').read_text())
        assert set(selection['selected'])==selected
        assert selection['primary_tasks_sha256']==hashlib.sha256((parent/'tasks.json').read_bytes()).hexdigest()
        return out
    out.mkdir(exist_ok=True)
    (out/'.runtime').mkdir(exist_ok=True)
    original=json.loads((parent/'tasks.json').read_text())
    tasks=copy.deepcopy(original)
    tasks['tasks']=[t for t in tasks['tasks'] if t['id'] in selected]
    for t in tasks['tasks']:
        t['max_tokens']=8192 if t['thinking'] else 3072
    save(out/'tasks.json',tasks)
    prepared=json.loads((parent/'prepared.json').read_text())
    prepared['tasks']=[t for t in prepared['tasks'] if t['id'] in selected]
    for t in prepared['tasks']:
        t['max_tokens']=8192 if t['thinking'] else 3072
    prepared['tasks_sha256']=hashlib.sha256((out/'tasks.json').read_bytes()).hexdigest()
    save(out/'prepared.json',prepared)
    for name in ['ko64k.pt','ko128k.pt','eagle_worker_v2.py']:
        shutil.copyfile(parent/'.runtime'/name,out/'.runtime'/name)
    shutil.copyfile(parent/'estimate.json',out/'estimate.json')
    shutil.copyfile(parent/'token-metadata.json',out/'token-metadata.json')
    (out/'.gitignore').write_text('.runtime/\n')
    save(out/'selection.json',dict(
        criterion='Any truncated or missing-final-answer request in the 48 primary 64K requests',
        selected=[t['id'] for t in tasks['tasks']],before_any_primary_128k_response=True,
        primary_tasks_sha256=hashlib.sha256((parent/'tasks.json').read_bytes()).hexdigest(),
        rounds=1,thinking_max_tokens=8192,other_max_tokens=3072,
        order='Each mode immediately after its primary requests, before unloading',
        purpose='Completion/quality follow-up; does not replace the frozen primary 96 requests'))
    print('Supplement frozen: '+', '.join(t['id'] for t in tasks['tasks']),flush=True)
    return out


def run(url, output, generate, incremental):
    prepared=json.loads((output.parent/'prepared.json').read_text())
    if output.exists() or output.with_suffix('.jsonl').exists():
        raise RuntimeError('Refusing to overwrite supplementary responses')
    flush(url)
    warm=generate(url,dict(prepared['tasks'][0],max_tokens=32),8182,incremental)
    save(output.with_name(output.stem+'-warmup.json'),warm)
    rows=[]
    with output.with_suffix('.jsonl').open('w',buffering=1) as stream:
        for task in prepared['tasks']:
            flush(url)
            result=generate(url,task,9182,incremental)
            row=dict(id=task['id'],domain=task['domain'],repeat=0,thinking=task['thinking'],
                     max_tokens=task['max_tokens'],input_tokens=len(task['input_ids']),**result)
            rows.append(row)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
            print(f'SUPPLEMENT {len(rows)}/{len(prepared["tasks"])} {task["id"]} '
                  f'tokens={len(row["output_ids"])} elapsed={row["elapsed_s"]:.1f}s '
                  f'truncated={row["truncated"]}',flush=True)
    save(output,dict(tasks_sha256=prepared['tasks_sha256'],rounds=1,rows=rows))
