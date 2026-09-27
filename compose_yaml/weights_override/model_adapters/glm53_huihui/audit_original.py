"""Stream the pinned original once; retain hashes, not another full checkpoint."""
import hashlib,json,math,os,time,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from inspect_headers import get_range
SIZES={0:(1,4),1:(1,2),8:(32,34),12:(256,144),13:(256,176),14:(256,210),30:(1,2)}
def nbytes(t):
    block,size=SIZES[t['type']];n=math.prod(t['shape'])
    if n%block:raise ValueError('Unaligned quantization block')
    return n//block*size

def audit_file(header_path,out):
    while not header_path.exists():time.sleep(5)
    h=json.loads(header_path.read_text());dst=out/(header_path.stem+'.hashes.json')
    report=json.loads(dst.read_text()) if dst.exists() else {'revision':h['revision'],'repo':h['repo'],'file':h['file'],'status':'running','tensors':[]}
    if report['revision']!=h['revision']:raise ValueError('Revision mismatch')
    done={t['name'] for t in report['tensors']};pending=[t for t in sorted(h['tensors'],key=lambda t:t['offset']) if t['name'] not in done]
    def save():
        tmp=dst.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(dst)
    errors=0
    while pending:
        offset=pending[0]['offset'];end=h['size']-1
        try:
            # Bounded parallel ranges avoid long CDN streams that stall mid-tensor.
            with ThreadPoolExecutor(max_workers=8) as pool:
                positions=iter(range(offset,end+1,4<<20));queue=deque()
                def submit():
                    pos=next(positions,None)
                    if pos is not None:queue.append(pool.submit(get_range,h['repo'],h['revision'],h['file'],pos,min(4<<20,end+1-pos)))
                for _ in range(8):submit()
                buf=b'';cursor=0
                def take(n):
                    nonlocal buf,cursor
                    result=bytearray()
                    while n:
                        if cursor==len(buf):
                            if not queue:raise EOFError('Short range stream')
                            buf=queue.popleft().result();cursor=0;submit()
                        k=min(n,len(buf)-cursor);result.extend(buf[cursor:cursor+k]);cursor+=k;n-=k
                    return result
                for t in list(pending):
                    gap=t['offset']-offset
                    if gap<0:raise ValueError('Overlapping tensors')
                    take(gap)
                    length=nbytes(t);remaining=length;digest=hashlib.sha256()
                    while remaining:
                        n=min(4<<20,remaining);digest.update(take(n));remaining-=n
                    report['tensors'].append({**t,'bytes':length,'sha256':digest.hexdigest()});save()
                    offset=t['offset']+length;pending.pop(0);errors=0
                    if len(report['tensors'])%20==0:print(h['file'],len(report['tensors']),'/',len(h['tensors']),flush=True)
        except Exception as e:
            errors+=1;print('RETRY',h['file'],type(e).__name__,str(e)[:180],flush=True)
            if errors>=8:raise
            time.sleep(min(30,2**errors))
    report['status']='complete';save();print('VERIFIED ORIGINAL',h['file'],flush=True)
def run(headers,out):
    out.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i:audit_file(headers/f'original-GLM-5.3-Flash-UD-Q4_K_XL-{i:05d}-of-00006.gguf.json',out),range(1,7)))
if __name__=='__main__':
    import sys
    run(Path(sys.argv[1]),Path(sys.argv[2]))
