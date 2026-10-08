"""Fetch pinned original GGUF changed tensor ranges, validating prior full audit hashes."""
import argparse,hashlib,json,time,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

def fetch_one(r,repo,rev,output,side):
 dest=output/(r['name']+'.q8');expected=r[side+'_sha256']
 if dest.exists():
  with dest.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
  return r['name']
 offset=r[side+'_offset'];length=r['bytes']
 part=dest.with_suffix('.partial.'+str(os.getpid()))
 for attempt in range(5):
  try:
   url=f'https://huggingface.co/{repo}/resolve/{rev}/{r["pair"]}?download=true&t={time.time_ns()}'
   with requests.get(url,headers={'Range':f'bytes={offset}-{offset+length-1}'},stream=True,timeout=(30,180)) as res:
    res.raise_for_status()
    if res.status_code!=206 or not res.headers.get('Content-Range','').startswith(f'bytes {offset}-{offset+length-1}/'):raise ValueError('Server did not honor exact range')
    h=hashlib.sha256();n=0
    with part.open('wb') as f:
     for block in res.iter_content(4<<20):f.write(block);h.update(block);n+=len(block)
    if n!=length or h.hexdigest()!=expected:raise ValueError('Range checksum/size mismatch')
   part.replace(dest);break
  except Exception:
   if attempt==4:raise
   time.sleep(2**attempt)
 print('VERIFIED',r['name'],r['bytes'],flush=True)
 return r['name']

def run(audit,output,workers=1,side="original"):
 report=json.loads(audit.read_text());assert report['status']=='complete'
 repo,rev=report['sources'][side];output.mkdir(parents=True,exist_ok=True)
 records=[r for r in report['tensors'] if r['changed']]
 assert all(r['type']=='Q8_0' for r in records)
 with ThreadPoolExecutor(max_workers=workers) as pool:
  done=list(pool.map(lambda r:fetch_one(r,repo,rev,output,side),records))
 status={'status':'complete','completed':done,'repo':repo,'revision':rev,'audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest()}
 temp=output/('status.'+str(os.getpid())+'.tmp');temp.write_text(json.dumps(status,indent=2));temp.replace(output/'status.json')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--side',choices=['original','huihui'],default='original');a=p.parse_args();run(a.audit,a.output,a.workers,a.side)
