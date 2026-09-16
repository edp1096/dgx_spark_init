#!/usr/bin/env python3
"""Transfer an audited GGUF delta onto RadixArk BF16 tensors; preserve NVFP4 experts."""
import argparse,hashlib,json,math,os,re,shutil,subprocess,time
from pathlib import Path
import numpy as np
import gguf

RADIX_REPO='RadixArk/Qwen3.8-Flash-Next-NVFP4'
RADIX_REV='7b719225242aacd3dbd3f9407468c2ee9a9d2594'
MAPPING={'ssm_out.weight':'linear_attn.out_proj.weight','attn_output.weight':'self_attn.o_proj.weight','ffn_down_shexp.weight':'mlp.shared_expert.down_proj.weight'}

# Also works in the offline container with /weights_core mounted read-only.
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "weights_core").is_dir())))
from weights_core.safetensors_io import read_header
from weights_core.numeric import bf16_to_float, float_to_bf16, norm2

def restore_gdn_columns(values, rows, config):
 # Inverse of llama.cpp _LinearAttentionVReorderBase._reorder_v_heads(dim=1).
 # Pinned converter: 391fac16460f15233a7740550d858ac96df3419d/conversion/qwen.py.
 c=config['text_config'];k=c['linear_num_key_heads'];v=c['linear_num_value_heads'];d=c['linear_value_head_dim']
 assert v%k==0 and values.size==rows*v*d
 return values.reshape(rows,v//k,k,d).transpose(0,2,1,3).copy().reshape(-1)

def mapped_name(record):
 m=re.fullmatch(r'blk\.(\d+)\.(.+)',record['name'])
 if not m or m[2] not in MAPPING or record['type']!='Q8_0':raise ValueError('Unsupported changed tensor: '+record['name'])
 return f'model.language_model.layers.{m[1]}.{MAPPING[m[2]]}'

def build(cache,audit_path,output,image,source=None,independent=False):
 report=json.loads(audit_path.read_text());assert report['status']=='complete'
 source=source or cache/'hub'/('models--'+RADIX_REPO.replace('/','--'))/'snapshots'/RADIX_REV
 config=json.loads((source/'config.json').read_text());assert config['quantization_config']['quant_algo']=='NVFP4'
 index=json.loads((source/'model.safetensors.index.json').read_text());weights=index['weight_map']
 all_changes=[t for t in report['tensors'] if t['changed']]
 expert_changes=[t for t in all_changes if t['name'].endswith('.ffn_down_exps.weight')]
 changes=[t for t in all_changes if t not in expert_changes]
 for t in expert_changes:
  assert t['type']=='Q8_0' and t['shape']==[640,2560,512], 'Unsupported expert layout'
 assert changes,'No differences to transfer'
 plan=[];headers={}
 for record in changes:
  name=mapped_name(record);shard=weights[name]
  if shard not in headers:headers[shard]=read_header(source/shard)
  header,start=headers[shard];tensor=header[name]
  if tensor['dtype']!='BF16' or tensor['shape']!=list(reversed(record['shape'])):raise ValueError('HF tensor layout mismatch: '+name)
  if math.prod(tensor['shape'])*2!=tensor['data_offsets'][1]-tensor['data_offsets'][0]:raise ValueError('Invalid BF16 length')
  plan.append((record,name,shard,tensor,start))
 work=output.with_name('.'+output.name+'.partial')
 if output.exists() or work.exists():raise FileExistsError('Refusing to overwrite candidate or partial directory')
 work.mkdir(parents=True)
 manifest={'status':'building','independent_copy':independent,'method':'BF16_original + dequant(Huihui_Q8) - dequant(Unsloth_Q8), rounded BF16 RNE','source':RADIX_REPO,'source_revision':RADIX_REV if source.name==RADIX_REV else None,'source_path':str(source),'audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),'gguf_sources':report['sources'],'expert_layers_pending':[int(t['name'].split('.')[1]) for t in expert_changes],'modified_shards':sorted(headers),'tensors':[],'started_at':time.time()}
 def save(): (work/'transfer-manifest.json').write_text(json.dumps(manifest,indent=2))
 save();print('PLAN',len(plan),'tensors in',len(headers),'BF16 shards',flush=True)
 # Runtime metadata only: upstream qualification reports do not describe this derivative.
 runtime_names=['config.json','hf_quant_config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.jinja','preprocessor_config.json','processor_config.json','video_preprocessor_config.json','special_tokens_map.json','added_tokens.json','LICENSE']
 for name in runtime_names:
  if (source/name).is_file():shutil.copyfile(source/name,work/name)
 missing_links=[]
 for shard in sorted(set(weights.values())):
  original=(source/shard).resolve();target=work/shard
  if shard in headers or independent:
   print('COPY MODIFIED SHARD',shard,flush=True);shutil.copyfile(original,target)
  else:
   try:os.link(original,target)
   except PermissionError:missing_links.append((str(original.relative_to(cache)),str(target.relative_to(cache))))
 if missing_links:
  # Root-owned cache blobs cannot be hardlinked by an unprivileged UID. The
  # helper links only the enumerated immutable files within the single cache mount.
  (work/'links.json').write_text(json.dumps(missing_links))
  code="import json,os,pathlib;root=pathlib.Path('/hf');pairs=json.load(open(root/"+repr(str((work/'links.json').relative_to(cache)))+"));[(os.link(root/a,root/b)) for a,b in pairs]"
  subprocess.run(['docker','run','--rm','--network','none','--read-only','--user','0','--entrypoint','python3','--mount',f'type=bind,src={cache},dst=/hf',image,'-c',code],check=True)
  (work/'links.json').unlink()
 pairs={p['label']:p for p in report['pairs']}
 for record,name,shard,tensor,start in plan:
  pair=pairs[record['pair']];rows,cols=tensor['shape'];assert cols%32==0
  ggrow=cols//32*34;baseoffset=start+tensor['data_offsets'][0]
  hashes=[hashlib.sha256(),hashlib.sha256()];metrics={'original_norm2':0.,'delta_norm2':0.,'original_q8_error2':0.,'rounding_error2':0.,'changed_values':0}
  with open(pair['original'],'rb') as a,open(pair['huihui'],'rb') as b,(source/shard).open('rb') as base,(work/shard).open('r+b') as dest:
   a.seek(record['original_offset']);b.seek(record['huihui_offset']);base.seek(baseoffset);dest.seek(baseoffset)
   for first in range(0,rows,128):
    n=min(128,rows-first);ab=a.read(n*ggrow);bb=b.read(n*ggrow);raw=base.read(n*cols*2)
    assert len(ab)==len(bb)==n*ggrow and len(raw)==n*cols*2
    hashes[0].update(ab);hashes[1].update(bb)
    qa=gguf.dequantize(np.frombuffer(ab,dtype=np.uint8),gguf.GGMLQuantizationType.Q8_0).reshape(-1)
    qb=gguf.dequantize(np.frombuffer(bb,dtype=np.uint8),gguf.GGMLQuantizationType.Q8_0).reshape(-1)
    if record['name'].endswith('.ssm_out.weight'):
     qa=restore_gdn_columns(qa,n,config);qb=restore_gdn_columns(qb,n,config)
    original=bf16_to_float(raw);delta=qb-qa;result=original+delta;encoded=float_to_bf16(result);dest.write(encoded)
    metrics['original_norm2']+=norm2(original);metrics['delta_norm2']+=norm2(delta);metrics['original_q8_error2']+=norm2(original-qa);metrics['rounding_error2']+=norm2(bf16_to_float(encoded)-result)
    metrics['changed_values']+=int(np.count_nonzero(np.frombuffer(encoded,dtype='<u2')!=np.frombuffer(raw,dtype='<u2')))
  assert hashes[0].hexdigest()==record['original_sha256'] and hashes[1].hexdigest()==record['huihui_sha256'],'GGUF audit inputs changed'
  denominator=max(metrics['original_norm2'],1e-30)
  metrics['original_q8_relative_error']=math.sqrt(metrics['original_q8_error2']/denominator)
  metrics['delta_relative_norm']=math.sqrt(metrics['delta_norm2']/denominator)
  if metrics['original_q8_relative_error']>.08:raise ValueError('Source/quantization alignment requires investigation: '+name)
  manifest['tensors'].append({'gguf_name':record['name'],'hf_name':name,'shard':shard,'shape':tensor['shape'],**metrics});save()
  print('PATCH',name,'delta',round(metrics['delta_relative_norm'],6),'source_Q8_error',round(metrics['original_q8_relative_error'],6),flush=True)
 for shard in set(weights.values())-headers.keys():
  assert os.path.samefile(source/shard,work/shard)==(not independent),'Unexpected unmodified shard storage'
 for shard in headers:
  assert not os.path.samefile(source/shard,work/shard),'Modified file must be independent'
  assert (source/shard).stat().st_size==(work/shard).stat().st_size
  assert read_header(source/shard)==read_header(work/shard)
 manifest.update(status='dense_complete' if expert_changes else 'complete',finished_at=time.time(),unchanged_shards=len(set(weights.values())-headers.keys()))
 save()
 (work/'README.md').write_text('# Qwen3.8 Flash Next — Huihui GGUF-delta / RadixArk NVFP4 derivative\n\nLocally derived; not an official Huihui release.\n\n'+manifest['method']+'\n\nRouted NVFP4 experts and all other unmodified tensors retain RadixArk weights. Quantization residuals remain in the transferred delta. See transfer-manifest.json for provenance and numeric checks. Runtime/behavioral evaluation is separate.\n')
 if expert_changes:
  print('DENSE PATCH COMPLETE; expert requantization pending:',work,flush=True)
 else:
  work.rename(output);print('BUILT',output,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=Path.home()/'.cache/huggingface');p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--helper-image',default='dgx-sglang-qwen38-fn:sm121-b12x-head-v1');a=p.parse_args();build(a.cache,a.audit,a.output,a.helper_image)
