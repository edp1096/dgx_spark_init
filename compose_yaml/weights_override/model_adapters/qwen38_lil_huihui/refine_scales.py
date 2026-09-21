"""Refine an UNQUALIFIED, independently copied candidate using original QAD scales.

For each changed NVFP4 expert matrix, retain the lower-error reconstruction of
fresh scales and original scales. Never touch a runtime-qualified artifact.
Hash every byte outside permitted expert tensor ranges before and after.
"""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import gguf
from model_adapters.qwen38_lil_huihui.build import decode,read,blob,Checkpoint,NVFP4QTensor

def hash_regions(path,intervals):
 full=hashlib.sha256();outside=hashlib.sha256();cursor=0
 with path.open('rb') as f:
  for begin,end in sorted(intervals)+[(path.stat().st_size,path.stat().st_size)]:
   assert begin>=cursor
   for length,untouched in [(begin-cursor,True),(end-begin,False)]:
    while length:
     block=f.read(min(length,8<<20));assert block;full.update(block)
     if untouched:outside.update(block)
     length-=len(block)
   cursor=end
 return {'full':full.hexdigest(),'outside':outside.hexdigest()}

def run(a):
 torch.set_num_threads(2);cp=Checkpoint(a.base);candidate=Checkpoint(a.candidate)
 assert cp.tensors==candidate.tensors
 mp=a.candidate/'transfer-manifest.json';original_manifest=mp.read_bytes();m=json.loads(original_manifest)
 assert m['status']=='candidate_verified' and not m.get('runtime_validated')
 assert not (a.candidate/'runtime-qualification.json').exists()
 report=json.loads(a.audit.read_text());records=[r for r in report['tensors'] if r['changed'] and r['name'].endswith('.ffn_down_exps.weight')]
 assert len(records)==5
 intervals={}
 for r in records:
  layer=int(r['name'].split('.')[1])
  for e in range(512):
   for suffix in ['weight','weight_scale','weight_scale_2']:
    name=f'model.language_model.layers.{layer}.mlp.experts.{e}.down_proj.{suffix}';t=cp.tensors[name]
    assert name in m['changed_tensors'];assert not os.path.samefile(a.base/t['shard'],a.candidate/t['shard'])
    intervals.setdefault(t['shard'],[]).append(tuple(t['start']+x for x in t['data_offsets']))
 before={f:hash_regions(a.candidate/f,spans) for f,spans in intervals.items()}
 # Validate source and destination shards used in this refinement before writing.
 for f in intervals:
  assert before[f]['full']==m['output_shard_hashes'][f]
  with (a.base/f).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==m['source_shard_hashes'][f]
 previous=a.candidate/'initial-transfer-manifest.json';previous.write_bytes(original_manifest)
 m['status']='refining';mp.write_text(json.dumps(m,indent=2));results=[]
 for r in records:
  layer=int(r['name'].split('.')[1]);stats={k:0. for k in ['base_norm2','delta_norm2','target_norm2','error_norm2','effective_change_norm2','delta_dot_effective']};stats.update(matrices=0,original_scales_selected=0,fresh_error_norm2=0.)
  ha,hb=hashlib.sha256(),hashlib.sha256()
  with (a.original/(r['name']+'.q8')).open('rb') as fa,(a.donor/(r['name']+'.q8')).open('rb') as fb:
   for e in range(512):
    prefix=f'model.language_model.layers.{layer}.mlp.experts.{e}.down_proj';value,_=decode(cp,prefix);current,_=decode(candidate,prefix);n=value.numel()//32*34
    ra,rb=fa.read(n),fb.read(n);assert len(ra)==len(rb)==n;ha.update(ra);hb.update(rb)
    qa=torch.from_numpy(gguf.dequantize(np.frombuffer(ra,dtype=np.uint8),gguf.GGMLQuantizationType.Q8_0).copy()).reshape(value.shape)
    qb=torch.from_numpy(gguf.dequantize(np.frombuffer(rb,dtype=np.uint8),gguf.GGMLQuantizationType.Q8_0).copy()).reshape(value.shape)
    delta=qb-qa;target=value+delta;s=read(cp,prefix+'.weight_scale');g=read(cp,prefix+'.weight_scale_2')
    q,fs,fg=NVFP4QTensor.quantize(target.to(torch.bfloat16),16,weights_scaling_factor=s,weights_scaling_factor_2=g)
    fixed=q.dequantize(dtype=torch.float32,scale=fs,double_scale=fg,block_sizes={-1:16});assert torch.isfinite(fixed).all()
    err=lambda x:float(torch.sum((x-target).double().square()))
    fresh_error=err(current);fixed_error=err(fixed);stats['fresh_error_norm2']+=fresh_error
    if fixed_error<fresh_error:
     for suffix,x in {'weight':q._quantized_data,'weight_scale':fs,'weight_scale_2':fg}.items():
      t=cp.tensors[prefix+'.'+suffix];b=blob(x);assert list(x.shape)==t['shape'] and len(b)==t['data_offsets'][1]-t['data_offsets'][0]
      with (a.candidate/t['shard']).open('r+b') as out:out.seek(t['start']+t['data_offsets'][0]);out.write(b)
     current=fixed;stats['original_scales_selected']+=1
    for key,x in [('base_norm2',value),('delta_norm2',delta),('target_norm2',target),('error_norm2',target-current),('effective_change_norm2',current-value)]:stats[key]+=float(torch.sum(x.double().square()))
    stats['delta_dot_effective']+=float(torch.sum(delta.double()*(current-value).double()));stats['matrices']+=1
    if e%64==0:print('REFINE',layer,e,flush=True)
  assert ha.hexdigest()==r['original_sha256'] and hb.hexdigest()==r['huihui_sha256']
  for key,num,den in [('requant_relative_l2','error_norm2','target_norm2'),('delta_relative_l2','delta_norm2','base_norm2'),('effective_change_relative_l2','effective_change_norm2','base_norm2')]:stats[key]=math.sqrt(stats[num]/stats[den])
  stats['delta_effective_cosine']=stats['delta_dot_effective']/max(math.sqrt(stats['delta_norm2']*stats['effective_change_norm2']),1e-30)
  assert stats['error_norm2']<=stats['fresh_error_norm2']
  row={'gguf_name':r['name'],**stats};results.append(row);print('REFINED',json.dumps(row),flush=True)
  (a.candidate/'scale-refinement-progress.json').write_text(json.dumps(results,indent=2))
 assert cp.tensors==Checkpoint(a.candidate).tensors
 for f,spans in intervals.items():
  after=hash_regions(a.candidate/f,spans)
  assert after['outside']==before[f]['outside'],'Unexpected bytes changed outside expert targets'
  m['output_shard_hashes'][f]=after['full']
 rows={r['gguf_name']:r for r in results};m['tensors']=[rows.get(r['gguf_name'],r) for r in m['tensors']]
 m.update(status='candidate_verified',scale_policy='NVFP4 expert matrices: minimum Frobenius reconstruction error of refreshed versus original LIL scales; MXFP8 refreshed',scale_refinement={'converter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'initial_manifest_sha256':hashlib.sha256(original_manifest).hexdigest(),'complement_sha256':before,'finished':time.time()})
 mp.write_text(json.dumps(m,indent=2));print('REFINEMENT VERIFIED',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for x in ['base','candidate','original','donor','audit']:p.add_argument('--'+x,type=Path,required=True)
 run(p.parse_args())
