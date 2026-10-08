"""Fetch only unresolved BF16 tensors after a complete bounded Xet audit.

Requires an explicit network-byte budget. Reuses all available earlier local
shard prefixes. Never falls back to a full model or ignores a range response.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests
from fetch_audit import atomic_json, manifest


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def local_prefix(root,side,row):
    path=root/'audit'/'scratch'/side/row['shard']
    options=[p for p in [path,path.with_suffix(path.suffix+'.partial')] if p.exists()]
    if not options:return None,0
    path=max(options,key=lambda p:p.stat().st_size)
    start=row[side+'_offset']
    return path,max(0,min(row['bytes'],path.stat().st_size-start))


def remaining(root,side,row):
    dest=root/'audit'/side/(row['name']+'.bf16')
    if dest.exists():return 0
    partial=dest.with_suffix('.partial')
    done=partial.stat().st_size if partial.exists() else 0
    _,available=local_prefix(root,side,row)
    return max(0,row['bytes']-max(done,available))


def fetch(root,side,row,transport='range'):
    dest=root/'audit'/side/(row['name']+'.bf16')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if dest.stat().st_size!=row['bytes']:raise ValueError('Existing tensor has wrong size')
        return digest(dest)
    partial=dest.with_suffix('.partial')
    done=partial.stat().st_size if partial.exists() else 0
    if done>row['bytes']:raise ValueError('Partial tensor too long')
    path,available=local_prefix(root,side,row)
    if available>done:
        with path.open('rb') as f,partial.open('ab') as out:
            f.seek(row[side+'_offset']+done)
            while done<available:
                block=f.read(min(4<<20,available-done))
                if not block:raise ValueError('Truncated reusable prefix')
                out.write(block);done+=len(block)
    api,files=manifest(root,side)
    record=files[row['shard']]
    if transport=='xet':
        from xet_ranges import XetRanges
        reader=XetRanges(root,side)
        log=root/'audit'/'transport'/(hashlib.sha256(row['name'].encode()).hexdigest()+'.'+side+'.json')
        log.parent.mkdir(parents=True,exist_ok=True)
        start_done=done
        while done<row['bytes']:
            end=min(row['bytes'],done+(64<<20))
            block=reader.range(row['shard'],row[side+'_offset']+done,row[side+'_offset']+end)
            if len(block)!=end-done:raise ValueError('Incomplete lossless reconstruction')
            with partial.open('ab') as out:out.write(block)
            done=end
            atomic_json(log,{'transport':'lossless_xet','starting_bytes':start_done,
                'decoded_bytes_written':done-start_done,'compressed_payload_bytes':reader.network_bytes})
        h=digest(partial);partial.replace(dest)
        return h
    with requests.Session() as session:
        while done<row['bytes']:
            end=min(row['bytes'],done+(64<<20))
            for attempt in range(5):
                try:
                    # Retry starts at the already-written byte; no duplicate payload.
                    done=partial.stat().st_size if partial.exists() else 0
                    if done==end:break
                    start_abs=row[side+'_offset']+done;end_abs=row[side+'_offset']+end
                    url=f"https://huggingface.co/{api['id']}/resolve/{api['sha']}/{row['shard']}?tensor_range={start_abs}-{end_abs}"
                    with session.get(url,headers={'Range':f'bytes={start_abs}-{end_abs-1}'},stream=True,timeout=(30,120)) as response:
                        response.raise_for_status()
                        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start_abs}-{end_abs-1}/{record["size"]}':
                            raise ValueError('Exact range not honored; refusing body')
                        with partial.open('ab') as out:
                            for block in response.iter_content(4<<20):
                                if done+len(block)>end:raise ValueError('Response exceeds requested range')
                                out.write(block);done+=len(block)
                    if done!=end:raise ValueError('Truncated range response')
                    break
                except (requests.RequestException,OSError) as exc:
                    if attempt==4:raise
                    print('RANGE RETRY',side,row['name'],attempt,type(exc).__name__,flush=True)
                    time.sleep(min(20,2**attempt))
    if partial.stat().st_size!=row['bytes']:raise ValueError('Incomplete tensor')
    h=digest(partial);partial.replace(dest)
    return h


def pair(root,row,transport='range'):
    receipt=root/'audit'/'selected-receipts'/(hashlib.sha256(row['name'].encode()).hexdigest()+'.json')
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        for side in ['original','huihui']:
            p=root/'audit'/side/(row['name']+'.bf16')
            if digest(p)!=saved[side+'_sha256']:raise ValueError('Previously saved tensor changed')
        return saved
    hashes={side:fetch(root,side,row,transport) for side in ['original','huihui']}
    saved=dict(row,changed=hashes['original']!=hashes['huihui'],**{s+'_sha256':h for s,h in hashes.items()})
    receipt.parent.mkdir(parents=True,exist_ok=True);atomic_json(receipt,saved)
    print('TENSOR VERIFIED',row['name'],'changed',saved['changed'],flush=True)
    return saved


def run(args):
    report=json.loads((args.root/'audit'/'xet-audit.json').read_text())
    if report['status']!='metadata_audit_complete':raise ValueError('Complete metadata audit required')
    needed=[r for r in report['tensors'] if r['status']!='equal']
    network=sum(remaining(args.root,s,r) for r in needed for s in ['original','huihui'])
    plan={'required_tensors':len(needed),'both_sides_bytes':sum(r['bytes'] for r in needed)*2,'new_network_bytes':network}
    atomic_json(args.root/'audit'/'selected-download-plan.json',plan)
    print(json.dumps(plan),flush=True)
    if args.plan_only:return
    if args.max_network_gib is None or network>args.max_network_gib*2**30:
        raise ValueError('Explicit network budget missing or insufficient; no tensor download started')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fetched=list(pool.map(lambda row:pair(args.root,row,args.transport),needed))
    by_name={r['name']:r for r in fetched}
    rows=[by_name[r['name']] if r['name'] in by_name else dict(r,changed=False) for r in report['tensors']]
    final=dict(status='complete',sources=report['sources'],method=report['method']+'; unresolved tensors compared by downloaded byte hashes',
               xet_audit_sha256=digest(args.root/'audit'/'xet-audit.json'),changed_count=sum(r['changed'] for r in rows),tensors=rows)
    atomic_json(args.root/'audit'/'tensor-audit.json',final)
    print('SELECTIVE AUDIT COMPLETE',len(rows),'changed',final['changed_count'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--max-network-gib',type=float)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--plan-only',action='store_true')
    p.add_argument('--transport',choices=['range','xet'],default='range')
    run(p.parse_args())
