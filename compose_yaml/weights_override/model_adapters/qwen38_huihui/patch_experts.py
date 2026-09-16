#!/usr/bin/env python3
"""Requantize only changed expert down projections using pinned ModelOpt NVFP4."""
import argparse,hashlib,json,math,os,shutil,time
from pathlib import Path
import numpy as np
import torch
import gguf
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from build import RADIX_REPO,RADIX_REV,read_header,bf16_to_float,float_to_bf16,norm2
from audit import snapshot

def raw(tensor):return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

def run(cache,audit,source_raw,output,probe_only=False,source=None):
 torch.set_num_threads(2)
 report=json.loads(audit.read_text());assert report['status']=='complete'
 records=[t for t in report['tensors'] if t['changed'] and t['name'].endswith('.ffn_down_exps.weight')]
 source=source or cache/'hub'/('models--'+RADIX_REPO.replace('/','--'))/'snapshots'/RADIX_REV
 work=output.with_name('.'+output.name+'.partial')
 status=json.loads((source_raw/'status.json').read_text());assert status['status']=='complete'
 index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
 results=[];headers={};copied=set()
 if not probe_only:
  manifest=json.loads((work/'transfer-manifest.json').read_text());assert manifest['status']=='dense_complete'
 for record in records:
  layer=int(record['name'].split('.')[1]);file=source_raw/f'layer-{layer:02d}-down.bf16'
  assert file.stat().st_size==512*2560*640*2
  h=hashlib.sha256()
  with file.open('rb') as f:
   for block in iter(lambda:f.read(8<<20),b''):h.update(block)
  assert h.hexdigest()==status['tensors'][str(layer)]['sha256']
  pair=next(p for p in report['pairs'] if p['label']==record['pair'])
  ap=Path(pair['original']);bp=Path(pair['huihui'])
  totals={'source_norm2':0.,'source_q8_error2':0.,'delta_norm2':0.,'nvfp4_error2':0.,'target_norm2':0.,'original_requantized_equal':0,'original_requantized_differing_bytes':{},'experts':0};ha=hashlib.sha256();hb=hashlib.sha256()
  with file.open('rb') as src,ap.open('rb') as a,bp.open('rb') as b:
   a.seek(record['original_offset']);b.seek(record['huihui_offset'])
   for expert in range(4 if probe_only else 512):
    origin=src.read(2560*640*2);ab=a.read(2560*640//32*34);bb=b.read(len(ab));ha.update(ab);hb.update(bb)
    qa=gguf.dequantize(np.frombuffer(ab,dtype=np.uint8),gguf.GGMLQuantizationType.Q8_0).reshape(-1);qb=gguf.dequantize(np.frombuffer(bb,dtype=np.uint8),gguf.GGMLQuantizationType.Q8_0).reshape(-1)
    original=bf16_to_float(origin);delta=qb-qa;encoded=float_to_bf16(original+delta)
    value=torch.frombuffer(bytearray(encoded),dtype=torch.bfloat16).reshape(2560,640)
    q,scale,global_scale=NVFP4QTensor.quantize(value,16)
    dequant=q.dequantize(dtype=torch.float32,scale=scale,double_scale=global_scale,block_sizes={-1:16}).numpy().reshape(-1)
    final=value.float().numpy().reshape(-1)
    totals['source_norm2']+=norm2(original);totals['source_q8_error2']+=norm2(original-qa);totals['delta_norm2']+=norm2(delta);totals['target_norm2']+=norm2(final);totals['nvfp4_error2']+=norm2(final-dequant);totals['experts']+=1
    prefix=f'model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj.'
    tensors={'weight':q._quantized_data,'weight_scale':scale,'weight_scale_2':global_scale}
    if expert<4:
     oq,oscale,oglobal=NVFP4QTensor.quantize(torch.frombuffer(bytearray(origin),dtype=torch.bfloat16).reshape(2560,640),16)
     expected={'weight':raw(oq._quantized_data),'weight_scale':raw(oscale),'weight_scale_2':raw(oglobal)};equal=True
     stored_parts={}
     for suffix,blob in expected.items():
      name=prefix+suffix;shard=index[name]
      if shard not in headers:headers[shard]=read_header(source/shard)
      hd,start=headers[shard];entry=hd[name]
      with (source/shard).open('rb') as f:
       f.seek(start+entry['data_offsets'][0]);stored=f.read(len(blob));equal &= stored==blob
      stored_parts[suffix]=stored
      count=int(np.count_nonzero(np.frombuffer(stored,dtype=np.uint8)!=np.frombuffer(blob,dtype=np.uint8)))
      totals['original_requantized_differing_bytes'][suffix]=totals['original_requantized_differing_bytes'].get(suffix,0)+count
     totals['original_requantized_equal']+=int(equal)
     if probe_only and not equal:
      sg=torch.frombuffer(bytearray(stored_parts['weight_scale_2']),dtype=torch.float32).reshape(())
      pq,ps,pg=NVFP4QTensor.quantize(torch.frombuffer(bytearray(origin),dtype=torch.bfloat16).reshape(2560,640),16,weights_scaling_factor_2=sg)
      same_scale_counts={k:int(np.count_nonzero(np.frombuffer(raw(v),dtype=np.uint8)!=np.frombuffer(stored_parts[k],dtype=np.uint8))) for k,v in {'weight':pq._quantized_data,'weight_scale':ps,'weight_scale_2':pg}.items()}
      print('SCALE_DIAGNOSTIC',layer,expert,json.dumps({'original':sg.item(),'recomputed':oglobal.item(),'relative_difference':abs(sg.item()-oglobal.item())/sg.item(),'same_scale_differing_bytes':same_scale_counts}),flush=True)
    if not torch.isfinite(scale.float()).all() or not (scale.float()>0).all() or not torch.isfinite(global_scale).all() or not (global_scale>0).all():raise ValueError('Invalid NVFP4 scales')
    if not probe_only:
     for suffix,tensor in tensors.items():
      name=prefix+suffix;shard=index[name]
      if shard not in copied:
       target=work/shard
       if not manifest.get('independent_copy'):
        assert os.path.samefile(source/shard,target)
        temp=target.with_suffix('.copy');shutil.copyfile(source/shard,temp);temp.replace(target)
       copied.add(shard)
      if shard not in headers:headers[shard]=read_header(source/shard)
      hd,start=headers[shard];entry=hd[name];blob=raw(tensor)
      assert list(tensor.shape)==entry['shape'] and len(blob)==entry['data_offsets'][1]-entry['data_offsets'][0]
      with (work/shard).open('r+b') as f:f.seek(start+entry['data_offsets'][0]);f.write(blob)
    if expert%64==0:print('EXPERT',layer,expert,flush=True)
  if not probe_only:assert ha.hexdigest()==record['original_sha256'] and hb.hexdigest()==record['huihui_sha256']
  totals['source_q8_relative_error']=math.sqrt(totals['source_q8_error2']/totals['source_norm2']);totals['nvfp4_relative_error']=math.sqrt(totals['nvfp4_error2']/totals['target_norm2'])
  assert totals['source_q8_relative_error']<.08 and totals['nvfp4_relative_error']<.20,totals
  results.append({'layer':layer,**totals});print('LAYER',layer,json.dumps(totals),flush=True)
  if not probe_only:(work/'expert-transfer-progress.json').write_text(json.dumps(results,indent=2))
 if probe_only:
  print('PROBE_RESULTS',json.dumps(results),flush=True);return
 manifest.update(status='complete',expert_layers_pending=[],expert_layers=results,expert_quantizer='ModelOpt source 87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c NVFP4QTensor.quantize, block 16, max weight scales, BF16 inputs',activation_calibration='Original RadixArk input scales retained; no fresh activation calibration',modified_shards=sorted(set(manifest['modified_shards'])|copied),finished_at=time.time(),original_bf16_source=json.loads((source_raw/'source-manifest.json').read_text()))
 manifest['unchanged_shards']=len(set(index.values())-set(manifest['modified_shards']))
 manifest['gdn_layout']='Inverse grouped/tiled V-head column permutation (16 K heads, 3 V heads per K, dim 128), llama.cpp 391fac16460f15233a7740550d858ac96df3419d'
 for shard in set(index.values()):
  assert os.path.samefile(source/shard,work/shard)==(not manifest.get('independent_copy') and shard not in manifest['modified_shards'])
  assert read_header(source/shard)==read_header(work/shard)
 (work/'README.md').write_text('# Qwen3.8 Flash Next — Huihui GGUF delta / RadixArk NVFP4 derivative\n\nLocally derived; not an official Huihui release.\n\n96 BF16 tensors receive the dequantized Huihui minus Unsloth Q8 delta, with GDN columns restored to HF order. Routed expert down projections in layers 2, 4, 30, 46, 47 receive the same delta on original Qwen BF16 weights and are requantized to NVFP4 with ModelOpt. All other weights retain RadixArk contents. Original activation input scales are retained; no new activation calibration was performed.\n\nQuantization residuals remain in the transferred delta; equivalence to an unavailable Huihui BF16 checkpoint is not claimed. See transfer-manifest.json and separate runtime evaluation.\n')
 (work/'transfer-manifest.json').write_text(json.dumps(manifest,indent=2));work.rename(output);print('BUILT',output,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--source-raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--probe-only',action='store_true');a=p.parse_args();run(a.cache,a.audit,a.source_raw,a.output,a.probe_only)
