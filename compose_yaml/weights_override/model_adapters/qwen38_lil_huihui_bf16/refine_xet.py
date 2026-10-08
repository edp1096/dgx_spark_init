"""Resolve small boundary/alignment ambiguities without fetching whole tensors."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import xet_audit as xa
from fetch_audit import atomic_json


def gaps(length,covered):
    out=[];cursor=0
    for start,end in sorted(covered):
        if start<cursor:raise ValueError('Overlapping proof intervals')
        if start>cursor:out.append((cursor,start))
        cursor=end
    if cursor<length:out.append((cursor,length))
    return out


def refine(root,a,b,row):
    if row['status']=='equal':return row
    ca=a.gap(a.file(row['shard']),row['name']);cb=b.gap(b.file(row['shard']),row['name'])
    nodes={};pos=ca['gap_start']-ca['start']
    for n in ca['nodes']:
        nodes[(pos,n['size'])]=n['hash'];pos+=n['size']
    covered=[];pos=cb['gap_start']-cb['start']
    for n in cb['nodes']:
        if nodes.get((pos,n['size']))==n['hash']:covered.append((pos,pos+n['size']))
        pos+=n['size']
    missing=gaps(row['bytes'],covered)
    if sum(e-s for s,e in missing)>xa.MAX_BOUNDARY:return row
    receipts=[]
    for start,end in missing:
        aa=a.bytes(row['shard'],row['original_offset']+start,row['original_offset']+end)
        bb=b.bytes(row['shard'],row['huihui_offset']+start,row['huihui_offset']+end)
        receipts.append(dict(start=start,end=end,original_sha256=hashlib.sha256(aa).hexdigest(),huihui_sha256=hashlib.sha256(bb).hexdigest()))
        if aa!=bb:return dict(row,reason='differing_complement_bytes_confirmed',complement_checks=receipts)
    print('REFINED EQUAL',row['name'],'payload_MiB',round(row['bytes']/2**20,2),'checked_MiB',round(sum(e-s for s,e in missing)/2**20,3),flush=True)
    return dict(row,status='equal',reason='matching_xet_nodes_and_complement_bytes_equal',
                covered_intervals=covered,complement_checks=receipts)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    path=args.root/'audit'/'xet-audit.json';report=json.loads(path.read_text())
    if report['status']!='metadata_audit_complete':raise ValueError('Complete first audit required')
    a,b=xa.Remote(args.root,'original'),xa.Remote(args.root,'huihui')
    xa.network_reserved=sum(p.stat().st_size for p in (args.root/'audit/xet').rglob('*.range'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(lambda r:refine(args.root,a,b,r),report['tensors']))
    backup=path.with_name('xet-audit-before-refinement.json')
    if backup.exists():raise FileExistsError('Refinement already recorded')
    atomic_json(backup,report)
    needed=[r for r in rows if r['status']!='equal']
    report.update(tensors=rows,equal_tensors=len(rows)-len(needed),requires_bytes_tensors=len(needed),
                  requires_bytes_both_sides=2*sum(r['bytes'] for r in needed),boundary_refinement=True)
    atomic_json(path,report)
    print('REFINEMENT COMPLETE','needs_bytes',len(needed),'both_GiB',report['requires_bytes_both_sides']/2**30,flush=True)
