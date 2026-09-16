#!/usr/bin/env python3
"""Verify that only audited target tensors changed; hash all candidate shards."""
import argparse,hashlib,json,os,re,time
from pathlib import Path
from build import RADIX_REPO,RADIX_REV,read_header

def digest(f, length):
 h=hashlib.sha256()
 while length:
  b=f.read(min(length,8<<20));assert b,'Unexpected EOF';h.update(b);length-=len(b)
 return h.hexdigest()

def verify(cache,candidate,out,source=None,source_hashes=None):
 start=time.time();m=json.loads((candidate/'transfer-manifest.json').read_text());assert m['status']=='complete'
 source=source or cache/'hub'/('models--'+RADIX_REPO.replace('/','--'))/'snapshots'/RADIX_REV
 index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
 targets={t['hf_name'] for t in m['tensors']}
 for layer in m.get('expert_layers',[]):
  for expert in range(512):
   for suffix in ['weight','weight_scale','weight_scale_2']:
    targets.add(f"model.language_model.layers.{layer['layer']}.mlp.experts.{expert}.down_proj.{suffix}")
 found=set();changed=0;unchanged=0;shards={}
 for shard in sorted(set(index.values())):
  base=source/shard;dest=candidate/shard;assert base.stat().st_size==dest.stat().st_size
  ha,start_a=read_header(base);hb,start_b=read_header(dest);assert ha==hb and start_a==start_b
  if shard in m['modified_shards']:
   assert not os.path.samefile(base,dest)
   with base.open('rb') as f:assert digest(f,base.stat().st_size)==(source_hashes[shard] if source_hashes is not None else base.resolve().name),'Modified original HF blob'
   with base.open('rb') as a,dest.open('rb') as b:
    for name,t in ha.items():
     if name=='__metadata__':continue
     begin,end=t['data_offsets'];a.seek(start_a+begin);b.seek(start_b+begin)
     x=digest(a,end-begin);y=digest(b,end-begin)
     if name in targets:
      found.add(name);changed+=int(x!=y)
     else:
      assert x==y,('Unexpected changed tensor',name);unchanged+=1
  else:
   assert os.path.samefile(base,dest)==(not m.get('independent_copy'))
   if m.get('independent_copy'):
    with base.open('rb') as f:assert digest(f,base.stat().st_size)==source_hashes[shard],'Modified original source'
   unchanged+=sum(k!='__metadata__' for k in ha)
  with dest.open('rb') as f:sha=digest(f,dest.stat().st_size)
  if shard not in m['modified_shards']:
   assert sha==(source_hashes[shard] if source_hashes is not None else base.resolve().name),'Original HF blob hash mismatch'
  shards[shard]={'size':dest.stat().st_size,'sha256':sha,'modified':shard in m['modified_shards']}
  print('VERIFIED',shard,flush=True)
 assert found==targets,(len(found),len(targets))
 result={'status':'passed','seconds':time.time()-start,'target_tensors':len(targets),'target_tensors_with_changed_bytes':changed,'unchanged_tensors':unchanged,'modified_shards':len(m['modified_shards']),'shards':shards}
 out.write_text(json.dumps(result,indent=2));print('VERIFIED ALL',json.dumps({k:v for k,v in result.items() if k!='shards'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.cache,a.candidate,a.output)
