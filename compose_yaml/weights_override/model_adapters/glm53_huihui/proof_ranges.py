"""Prove equal ranges from Xet identities and fetch only the unproven complement."""
import hashlib,json,time,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from inspect_headers import SOURCES,get_range

def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if a>=b:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out

def plans(headers,proof,out):
    result=[]
    for i in range(2,7):
        name=f'GLM-5.3-Flash-UD-Q4_K_XL-{i:05d}-of-00006.gguf'
        a=json.loads((headers/('original-'+name+'.json')).read_text());b=json.loads((headers/('huihui-'+name+'.json')).read_text())
        bt={t['name']:t for t in b['tensors']}
        shifts={bt[t['name']]['offset']-t['offset'] for t in a['tensors']}
        if len(shifts)!=1:raise ValueError('GGUF offset shift is not uniform')
        shift=shifts.pop();sides={}
        for side in SOURCES:
            chunks=sorted(proof.glob(f'{side}-{i}-*.json'))
            data=[json.loads(p.read_text()) for p in chunks]
            expected=a if side=='original' else b
            if not data or merge([(r['start'],r['end']) for r in data])!=[[0,expected['size']]]:raise ValueError('Xet coverage incomplete')
            if any(r['revision']!=SOURCES[side][1] or r['file']!=expected['file'] or r['file_size']!=expected['size'] for r in data):raise ValueError('Xet source mismatch')
            sides[side]=[t for r in data for t in r['terms']]
        def key(t):return (t['hash'],t['range']['start'],t['range']['end'],t['unpacked_length'])
        donor={}
        for t in sides['huihui']:donor.setdefault(key(t),set()).add(t['offset'])
        shared=merge([(max(a['data_start'],t['offset']),min(a['size'],t['offset']+t['unpacked_length'])) for t in sides['original'] if t['offset']+shift in donor.get(key(t),set())])
        gaps=[];pos=a['data_start']
        for start,end in shared:
            if pos<start:gaps.append([pos,start])
            pos=max(pos,end)
        if pos<a['size']:gaps.append([pos,a['size']])
        rec={'file':a['file'],'file_index':i,'size':a['size'],'shift':shift,'shared':shared,'gaps':gaps,'proof_method':'equal Xet xorb hash + chunk interval + unpacked length at corresponding GGUF offsets'}
        result.append(rec);print('PROVEN',i,'shared_GiB',sum(y-x for x,y in shared)/2**30,'fetch_GiB',sum(y-x for x,y in gaps)/2**30,flush=True)
    payload=json.dumps({'sources':SOURCES,'files':result},indent=2)
    if out.exists() and json.loads(out.read_text())!=json.loads(payload):raise ValueError('Existing proof plan differs; use a new workspace')
    out.write_text(payload);return result

def fetch_gap(args,out):
    file,index,start,end=args;dest=out/f'{index}-{start}-{end}.bin';receipt=dest.with_suffix('.json')
    if dest.exists() and receipt.exists():
        m=json.loads(receipt.read_text())
        with dest.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        if dest.stat().st_size==end-start and h==m['sha256']:return
    tmp=dest.with_suffix('.partial');h=hashlib.sha256();offset=0
    if tmp.exists():
        offset=tmp.stat().st_size
        if offset>end-start:raise ValueError('Oversized partial gap')
        with tmp.open('rb') as f:
            while b:=f.read(4<<20):h.update(b)
    with tmp.open('ab') as f:
        for pos in range(start+offset,end,1<<20):
            b=get_range(*SOURCES['original'],file,pos,min(1<<20,end-pos));h.update(b);f.write(b);f.flush()
    tmp.replace(dest);rtemp=receipt.with_suffix('.json.tmp');rtemp.write_text(json.dumps({'file':file,'start':start,'end':end,'sha256':h.hexdigest()}));rtemp.replace(receipt)
    print('GAP VERIFIED',index,start,end,flush=True)
def run(headers,proof,out):
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'download.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=out/'proof-plan.json';records=plans(headers,proof,p)
    tasks=[]
    for r in records:
        for start,end in r['gaps']:
            for offset in range(start,end,32<<20):tasks.append((r['file'],r['file_index'],offset,min(end,offset+(32<<20))))
    with ThreadPoolExecutor(max_workers=1) as pool:list(pool.map(lambda task:fetch_gap(task,out),tasks))
    (out/'complete.json').write_text(json.dumps({'status':'complete','ranges':len(tasks),'bytes':sum(t[3]-t[2] for t in tasks),'plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
if __name__=='__main__':
    import sys
    run(*(Path(s) for s in sys.argv[1:]))
