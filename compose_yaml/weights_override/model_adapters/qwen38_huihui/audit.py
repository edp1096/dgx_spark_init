#!/usr/bin/env python3
"""Read-only full tensor comparison of the pinned Unsloth/Huihui GGUF pair."""
import argparse,collections,hashlib,json,os,time
from pathlib import Path
import gguf

REPOS={
 'original':('unsloth/Qwen3.8-Flash-Next-GGUF','38bb39ee97821de2c9009abb7e93950eec396e66'),
 'huihui':('huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF','7e3bfc316b880fefeb049596f11c49d6a18e05fb'),
}
def snapshot(cache,key):
 repo,rev=REPOS[key];return cache/('models--'+repo.replace('/','--'))/'snapshots'/rev

def hash_region(file,offset,length):
 file.seek(offset);digest=hashlib.sha256();remaining=length
 while remaining:
  block=file.read(min(8<<20,remaining))
  if not block:raise EOFError('Truncated tensor')
  digest.update(block);remaining-=len(block)
 if hasattr(os,'posix_fadvise'):os.posix_fadvise(file.fileno(),offset,length,os.POSIX_FADV_DONTNEED)
 return digest.hexdigest()

def field_hash(field):
 h=hashlib.sha256()
 for p in field.parts:h.update(p.tobytes())
 return h.hexdigest()

def audit(cache,out,original_dir=None,donor_dir=None):
 started=time.time();report={'sources':REPOS,'status':'running','started_at':started,'pairs':[],'tensors':[],'bytes_read':0}
 def save():
  out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(out)
 if original_dir is not None or donor_dir is not None:
  report['sources']={'original':{'path':str(original_dir)},'huihui':{'path':str(donor_dir)}}
 original_dir=original_dir or snapshot(cache,'original')
 donor_dir=donor_dir or snapshot(cache,'huihui')
 pairs=[]
 for i in range(1,5):
  name=f'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-{i:05d}-of-00004.gguf'
  pairs.append((name,original_dir/name,donor_dir/name))
 pairs.append(('vision',original_dir/'mmproj-BF16.gguf',donor_dir/'mmproj-model-bf16.gguf'))
 save();last=time.time()
 for label,original,huihui in pairs:
  a,b=gguf.GGUFReader(original),gguf.GGUFReader(huihui)
  at={t.name:t for t in a.tensors};bt={t.name:t for t in b.tensors}
  if at.keys()!=bt.keys():raise ValueError('Tensor names differ: '+label)
  meta=[]
  for name in a.fields.keys()|b.fields.keys():
   if name not in a.fields or name not in b.fields or field_hash(a.fields[name])!=field_hash(b.fields[name]):meta.append(name)
  report['pairs'].append({'label':label,'original':str(original),'huihui':str(huihui),'tensor_count':len(at),'metadata_differences':sorted(meta)})
  with original.open('rb') as af,huihui.open('rb') as bf:
   for name,ta in at.items():
    tb=bt[name]
    if ta.tensor_type!=tb.tensor_type or ta.shape.tolist()!=tb.shape.tolist() or ta.n_bytes!=tb.n_bytes:raise ValueError('Tensor layout differs: '+name)
    ha=hash_region(af,ta.data_offset,ta.n_bytes);hb=hash_region(bf,tb.data_offset,tb.n_bytes)
    record={'pair':label,'name':name,'type':ta.tensor_type.name,'shape':ta.shape.tolist(),'bytes':ta.n_bytes,'original_offset':ta.data_offset,'huihui_offset':tb.data_offset,'original_sha256':ha,'huihui_sha256':hb,'changed':ha!=hb}
    report['tensors'].append(record);report['bytes_read']+=2*ta.n_bytes
    if ha!=hb:print('CHANGED',name,ta.tensor_type.name,flush=True)
    if time.time()-last>15:
     save();print('PROGRESS',len(report['tensors']),round(report['bytes_read']/1e9,2),'GB read',round(time.time()-started),'seconds',flush=True);last=time.time()
  del a,b,at,bt
  save()
 changed=[t for t in report['tensors'] if t['changed']]
 report.update(status='complete',finished_at=time.time(),changed_count=len(changed),tensor_count=len(report['tensors']),changed_types=dict(collections.Counter(t['type'] for t in changed)))
 save();print('DONE',report['tensor_count'],'tensors;',len(changed),'changed;',round(time.time()-started),'seconds',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=Path.home()/'.cache/huggingface/hub');p.add_argument('--out',type=Path,required=True);args=p.parse_args();audit(args.cache,args.out)
