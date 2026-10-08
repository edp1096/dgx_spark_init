#!/usr/bin/env python3
"""Download only pinned BF16 expert tensor byte ranges, resumably."""
import argparse,concurrent.futures,hashlib,json,threading,time,urllib.request
from pathlib import Path

def fetch(manifest,out):
 out.mkdir(parents=True,exist_ok=True);lock=threading.Lock();state={'status':'downloading','source':manifest['repo'],'revision':manifest['revision'],'tensors':{},'started_at':time.time()}
 def save():
  with lock:
   p=out/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(out/'status.json')
 def one(t):
  dest=out/f"layer-{t['layer']:02d}-down.bf16";part=dest.with_suffix('.part')
  if not dest.exists():
   for attempt in range(8):
    have=part.stat().st_size if part.exists() else 0
    if have==t['bytes']:break
    assert have<t['bytes'];start=t['offset']+have;end=t['offset']+t['bytes']-1
    request=urllib.request.Request(t['url']+f'?tensor_range={start}-{end}',headers={'Range':f'bytes={start}-{end}'})
    try:
     with urllib.request.urlopen(request,timeout=120) as response,part.open('ab') as f:
      assert response.status==206 and response.headers['Content-Range'].startswith(f'bytes {start}-'),response.headers.get('Content-Range')
      while have<t['bytes']:
       data=response.read(min(8<<20,t['bytes']-have))
       if not data:raise EOFError('Incomplete response')
       f.write(data);have+=len(data);state['tensors'][str(t['layer'])]={'received':have,'total':t['bytes']};save()
     break
    except Exception as e:
     print('RETRY',t['layer'],attempt,type(e).__name__,flush=True)
     if attempt==7:raise
     time.sleep(10)
   assert part.stat().st_size==t['bytes'];part.rename(dest)
  h=hashlib.sha256()
  with dest.open('rb') as f:
   for block in iter(lambda:f.read(8<<20),b''):h.update(block)
  state['tensors'][str(t['layer'])]={'received':dest.stat().st_size,'total':t['bytes'],'sha256':h.hexdigest(),'file':dest.name,'status':'complete'};save();print('COMPLETE',dest.name,flush=True)
 save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,manifest['tensors']))
 state['status']='complete';state['finished_at']=time.time();save();(out/'source-manifest.json').write_text(json.dumps(manifest,indent=2));print('ALL SOURCE TENSORS READY',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();fetch(json.loads(a.manifest.read_text()),a.out)
