#!/usr/bin/env python3
"""Controlled broad-domain native-API workload. Records real token IDs and MTP counts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import requests

from compare import flush
from vocab_memory import save


def generate(url, task, seed, incremental=False):
    body = dict(input_ids=task['input_ids'], stream=True,
                sampling_params=dict(temperature=0, max_new_tokens=task['max_tokens'],
                                     sampling_seed=seed, skip_special_tokens=False))
    started = datetime.datetime.now().astimezone().isoformat()
    start = time.monotonic()
    first = last = None
    first_tokens = 0
    ids, parts, meta = [], [], {}
    with requests.post(url + '/generate', json=body, stream=True, timeout=(10, 900)) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            chunk = json.loads(line[6:])
            if chunk.get('error'):
                raise RuntimeError(chunk['error'])
            now = time.monotonic()
            new_ids = chunk.get('output_ids') or []
            meta = chunk.get('meta_info') or meta
            count = meta.get('completion_tokens', 0)
            if count > 0 and first is None:
                first = now
                first_tokens = count
            if count > len(ids):
                last = now
            if incremental:
                ids.extend(new_ids)
                if chunk.get('text'):
                    parts.append(chunk['text'])
            else:
                if new_ids:
                    ids = new_ids
                if chunk.get('text') is not None:
                    parts = [chunk['text']]
    elapsed = time.monotonic() - start
    count = meta.get('completion_tokens', 0)
    if not ids or count != len(ids):
        raise RuntimeError(f'Missing/inconsistent actual output token IDs: {count} vs {len(ids)}')
    if not meta.get('spec_verify_ct'):
        raise RuntimeError('Per-request speculative counts missing')
    reason = meta.get('finish_reason', {})
    if reason.get('type') == 'abort':
        raise RuntimeError('Server aborted request: ' + str(reason))
    return dict(started=started, finished=datetime.datetime.now().astimezone().isoformat(),
                content=''.join(parts), output_ids=ids, meta_info=meta,
                ttft_s=first-start if first else None, elapsed_s=elapsed,
                first_chunk_tokens=first_tokens,
                decode_tok_s=(count-first_tokens)/(last-first)
                if first and last and last>first and count>first_tokens else None,
                end_to_end_tok_s=count/elapsed, truncated=reason.get('type')=='length')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    prepared = json.loads((out.parent/'prepared.json').read_text())
    assert prepared['tasks_sha256'] == hashlib.sha256((out.parent/'tasks.json').read_bytes()).hexdigest()
    if out.exists() or out.with_suffix('.jsonl').exists():
        raise RuntimeError('Refusing to overwrite trial outputs')
    info = requests.get(args.url+'/get_server_info', timeout=10).json()
    incremental = info.get('incremental_streaming_output', False)
    supplement = None
    if out.parent.name == '2026-09-08-broad-vocab' and out.stem == 'ko128k-benchmark':
        from broad_supplement import prepare
        supplement = prepare(out.parent)
    warmup = dict(prepared['tasks'][0], max_tokens=32)
    flush(args.url)
    warm = generate(args.url, warmup, 8182, incremental)
    save(out.with_name(out.stem+'-warmup.json'), warm)
    resume=out.parent/f'resume-{out.stem.split("-",1)[0]}.json'
    rows=json.loads(resume.read_text())['rows'] if resume.exists() else []
    completed={(r['id'],r['repeat']) for r in rows}
    if len(completed)!=len(rows):
        raise RuntimeError('Duplicate resumed requests')
    if rows:
        assert json.loads(resume.read_text())['tasks_sha256']==prepared['tasks_sha256']
        print(f'Resuming after {len(rows)} completed requests',flush=True)
    with out.with_suffix('.jsonl').open('w', buffering=1) as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
        for repeat in range(args.rounds):
            sequence=prepared['tasks'] if repeat%2==0 else list(reversed(prepared['tasks']))
            for task in sequence:
                if (task['id'],repeat) in completed:
                    continue
                flush(args.url)
                result=generate(args.url, task, 9182+repeat, incremental)
                row=dict(id=task['id'], domain=task['domain'], repeat=repeat,
                         thinking=task['thinking'], max_tokens=task['max_tokens'],
                         input_tokens=len(task['input_ids']), **result)
                rows.append(row)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                stream.flush(); os.fsync(stream.fileno())
                print(f'{len(rows)}/{len(prepared["tasks"])*args.rounds} {row["id"]} '
                      f'round={repeat+1} tokens={len(row["output_ids"])} '
                      f'elapsed={row["elapsed_s"]:.1f}s decode={row["decode_tok_s"]} '
                      f'accept={row["meta_info"]["spec_accept_rate"]:.3f} '
                      f'truncated={row["truncated"]}',flush=True)
    save(out,dict(tasks_sha256=prepared['tasks_sha256'],rounds=args.rounds,rows=rows))
    if out.parent.name == '2026-09-08-broad-vocab' and out.stem == 'ko64k-benchmark':
        from broad_supplement import prepare
        supplement = prepare(out.parent)
    if supplement:
        from broad_supplement import run
        run(args.url, supplement/out.name, generate, incremental)


if __name__=='__main__':
    main()
