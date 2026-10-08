"""Apply audited Huihui Q8 deltas to LIL's reconstructed QAD weights, CPU only.

Preserves all untouched tensor bytes and the exact mixed-precision checkpoint layout.
Never overwrites a source or advertises a candidate as runtime-qualified.
"""
import argparse,hashlib,json,math,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
import gguf
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from weights_core.safetensors_io import Checkpoint,read_header
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.quantization.qtensor.mxfp8_tensor import MXFP8QTensor

MAPPING={'ssm_out.weight':'linear_attn.out_proj','attn_output.weight':'self_attn.o_proj','ffn_down_shexp.weight':'mlp.shared_expert.down_proj'}
DT={'U8':torch.uint8,'F8_E4M3':torch.float8_e4m3fn,'F32':torch.float32}

def blob(t):return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
def read(cp,name):
 t=cp.tensors[name];raw=b''.join(cp.blocks(name))
 return torch.frombuffer(bytearray(raw),dtype=DT[t['dtype']]).reshape(t['shape'])
def restore_columns(t,config):
 c=config['text_config'];k=c['linear_num_key_heads'];v=c['linear_num_value_heads'];d=c['linear_value_head_dim']
 assert v%k==0 and t.shape[1]==v*d
 return t.reshape(t.shape[0],v//k,k,d).transpose(1,2).contiguous().reshape(t.shape)
def decode(cp,prefix):
 w=read(cp,prefix+'.weight');s=read(cp,prefix+'.weight_scale')
 if w.dtype==torch.float8_e4m3fn:
  assert s.dtype==torch.uint8 and list(s.shape)==[w.shape[0],w.shape[1]//32]
  value=(w.float().reshape(w.shape[0],-1,32)*torch.exp2(s.float()-127)[...,None]).reshape(w.shape)
  return value,'mxfp8'
 assert w.dtype==torch.uint8 and s.dtype==torch.float8_e4m3fn
 g=read(cp,prefix+'.weight_scale_2');assert g.numel()==1 and g.item()>0
 codes=torch.stack((w&15,w>>4),dim=-1).reshape(w.shape[0],-1).long()
 levels=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
 value=(levels[codes].reshape(w.shape[0],-1,16)*(s.float()*g)[...,None]).reshape(w.shape[0],-1)
 return value,'nvfp4'
def encode(value,kind):
 # Match existing exporter inputs. Record BF16 rounding as part of conversion.
 v=value.to(torch.bfloat16)
 if kind=='mxfp8':
  q,s=MXFP8QTensor.quantize(v);parts={'weight':q._quantized_data,'weight_scale':s}
  restored=q.dequantize(dtype=torch.float32,scale=s)
 else:
  q,s,g=NVFP4QTensor.quantize(v,16);parts={'weight':q._quantized_data,'weight_scale':s,'weight_scale_2':g}
  restored=q.dequantize(dtype=torch.float32,scale=s,double_scale=g,block_sizes={-1:16})
 assert torch.isfinite(restored).all()
 return parts,restored

def run(a):
 torch.set_num_threads(2);start=time.time();cp=Checkpoint(a.base)
 config=json.loads((a.base/'config.json').read_text());assert config['quantization_config']['quant_algo']=='MIXED_PRECISION'
 assert config['architectures'] and set(config['architectures']) <= {'Qwen4ExpForConditionalGeneration','Qwen3_8FlashNextForConditionalGeneration'}
 assert all(config['text_config'][k]==v for k,v in {'hidden_size':2560,'num_hidden_layers':48,'num_experts':512}.items())
 for directory in [a.base,a.donor,a.original]:
  if a.output.resolve()==directory.resolve() or directory.resolve() in a.output.resolve().parents or a.output.resolve() in directory.resolve().parents:raise ValueError('Output must be separate from inputs')
 report=json.loads(a.audit.read_text());assert report['status']=='complete' and report['changed_count']==101
 records=[r for r in report['tensors'] if r['changed']]
 for r in records:
  for directory,side in [(a.original,'original'),(a.donor,'huihui')]:
   f=directory/(r['name']+'.q8')
   if side=='original' or f.is_file():
    assert f.stat().st_size==r['bytes']
    with f.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==r[side+'_sha256']
 print('INPUT RANGE HASHES VERIFIED',flush=True)
 work=a.output.with_name('.'+a.output.name+'.partial')
 if a.output.exists() or work.exists():raise FileExistsError('Candidate/partial exists; refusing overwrite')
 if shutil.disk_usage(a.output.parent).free<sum((a.base/f).stat().st_size for f in set(t['shard'] for t in cp.tensors.values()))+2**30:raise ValueError('Insufficient disk space')
 work.mkdir();manifest={'status':'building','runtime_validated':False,'method':'DQ(LIL QAD) + DQ(Huihui Q8) - DQ(Unsloth Q8); BF16 rounding; ModelOpt requantization','base_revision':a.base.name,'gguf_sources':report['sources'],'audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),'activation_scales':'preserved, no new calibration','tensors':[],'source_shard_hashes':{},'started':start,'quantizer_revision':'87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c','converter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 def save():
  p=work/'transfer-manifest.json';p.with_suffix('.tmp').write_text(json.dumps(manifest,indent=2));p.with_suffix('.tmp').replace(p)
 # Independent complete copy; hashes establish immutable source provenance.
 for f in sorted(set(t['shard'] for t in cp.tensors.values())):
  h=hashlib.sha256()
  with (a.base/f).open('rb') as src,(work/f).open('xb') as out:
   for b in iter(lambda:src.read(8<<20),b''):h.update(b);out.write(b)
  manifest['source_shard_hashes'][f]=h.hexdigest();save();print('COPIED',f,flush=True)
 for p in a.base.iterdir():
  if p.is_file() and not p.name.endswith('.safetensors') and p.name!='README.md':shutil.copyfile(p,work/p.name)
 changed=set()
 for r in records:
  assert r['type']=='Q8_0';layer=int(r['name'].split('.')[1]);suffix='.'.join(r['name'].split('.')[2:]);expert=suffix=='ffn_down_exps.weight'
  donor_range=a.donor/(r['name']+'.q8')
  donor=donor_range if donor_range.is_file() else a.donor/r['pair'];orig=a.original/(r['name']+'.q8')
  assert orig.stat().st_size==r['bytes']
  ha=hashlib.sha256();hb=hashlib.sha256();stats={'base_norm2':0.,'delta_norm2':0.,'target_norm2':0.,'error_norm2':0.,'effective_change_norm2':0.,'delta_dot_effective':0.,'matrices':0}
  with orig.open('rb') as fa,donor.open('rb') as fb:
   fb.seek(0 if donor_range.is_file() else r['huihui_offset'])
   for e in range(512 if expert else 1):
    prefix=f'model.language_model.layers.{layer}.'+(f'mlp.experts.{e}.down_proj' if expert else MAPPING[suffix])
    value,kind=decode(cp,prefix)
    expected=list(reversed(r['shape']))
    assert list(value.shape)==(expected[1:] if expert else expected) and torch.isfinite(value).all()
    n=value.numel()//32*34
    ra,rb=fa.read(n),fb.read(n);assert len(ra)==len(rb)==n;ha.update(ra);hb.update(rb)
    qa=torch.from_numpy(gguf.dequantize(np.frombuffer(ra,dtype=np.uint8),gguf.GGMLQuantizationType.Q8_0).copy()).reshape(value.shape)
    qb=torch.from_numpy(gguf.dequantize(np.frombuffer(rb,dtype=np.uint8),gguf.GGMLQuantizationType.Q8_0).copy()).reshape(value.shape)
    delta=qb-qa
    if suffix=='ssm_out.weight':delta=restore_columns(delta,config)
    target=value+delta;parts,restored=encode(target,kind)
    for key,x in parts.items():
     name=prefix+'.'+key;t=cp.tensors[name];b=blob(x)
     assert list(x.shape)==t['shape'] and len(b)==t['data_offsets'][1]-t['data_offsets'][0]
     assert x.dtype==DT[t['dtype']]
     with (work/t['shard']).open('r+b') as f:f.seek(t['start']+t['data_offsets'][0]);f.write(b)
     changed.add(name)
    for key,x in [('base_norm2',value),('delta_norm2',delta),('target_norm2',target),('error_norm2',target-restored),('effective_change_norm2',restored-value)]:stats[key]+=float(torch.sum(x.double().square()))
    stats['delta_dot_effective']+=float(torch.sum(delta.double()*(restored-value).double()))
    stats['matrices']+=1
    if expert and e%64==0:print('EXPERT',layer,e,flush=True)
  assert ha.hexdigest()==r['original_sha256'] and hb.hexdigest()==r['huihui_sha256'],'GGUF range hash mismatch'
  stats['delta_relative_l2']=math.sqrt(stats['delta_norm2']/stats['base_norm2'])
  stats['effective_change_relative_l2']=math.sqrt(stats['effective_change_norm2']/stats['base_norm2'])
  stats['delta_effective_cosine']=stats['delta_dot_effective']/max(math.sqrt(stats['delta_norm2']*stats['effective_change_norm2']),1e-30)
  stats['requant_relative_l2']=math.sqrt(stats['error_norm2']/stats['target_norm2']);assert stats['requant_relative_l2']<(.2 if expert else .08)
  manifest['tensors'].append({'gguf_name':r['name'],**stats});save();print('PATCHED',r['name'],stats['requant_relative_l2'],flush=True)
 # Verify every non-target tensor exactly, plus original shards unchanged.
 out=Checkpoint(work);assert cp.tensors==out.tensors
 for f,expected in manifest['source_shard_hashes'].items():
  with (a.base/f).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
  assert not os.path.samefile(a.base/f,work/f)
 for name in cp.tensors:
  if name not in changed:assert cp.digest(name)==out.digest(name),'Non-target tensor changed: '+name
 manifest.update(status='candidate_verified',changed_tensors=sorted(changed),finished=time.time(),output_shard_hashes={},metadata_sha256={})
 for p in a.base.iterdir():
  if p.is_file() and not p.name.endswith('.safetensors') and p.name!='README.md':
   assert p.read_bytes()==(work/p.name).read_bytes()
   manifest['metadata_sha256'][p.name]=hashlib.sha256((work/p.name).read_bytes()).hexdigest()
 for f in manifest['source_shard_hashes']:
  with (work/f).open('rb') as stream:manifest['output_shard_hashes'][f]=hashlib.file_digest(stream,'sha256').hexdigest()
 save();shutil.copyfile(a.audit,work/'gguf-audit.json')
 (work/'README.md').write_text('# Huihui delta / LIL QAD NVFP4 candidate\n\nNot an official release. Runtime unverified.\n\n'+manifest['method']+'\n\nOriginal LIL activation scales retained; no new activation calibration. GGUF quantization residuals and additional requantization error remain. Does not claim equivalence to Huihui BF16 or preservation of original QAD benchmark scores. See transfer-manifest.json.\n')
 work.rename(a.output);print('CANDIDATE VERIFIED',a.output,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for x in ['base','donor','original','audit','output']:p.add_argument('--'+x,type=Path,required=True)
 run(p.parse_args())
