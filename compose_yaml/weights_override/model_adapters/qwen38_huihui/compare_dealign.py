#!/usr/bin/env python3
"""Direct complete tensor comparison; shared inodes prove equality without rereading."""
import collections,hashlib,json,math,os,re,time
from pathlib import Path
import numpy as np
from build import read_header,bf16_to_float,norm2
root=Path.home()/'.cache/huggingface'
a=root/'hub/models--dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4/snapshots/be794b990578ef3031eccf9f28e675a289a09ee9'
b=root/'edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4'
base=root/'hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594'
out=Path(__file__).resolve().parent/'docs/dealign-comparison.json'
start=time.time();ia=json.loads((a/'model.safetensors.index.json').read_text())['weight_map'];ib=json.loads((b/'model.safetensors.index.json').read_text())['weight_map'];assert ia==ib
same=0;changed=[];shards=[];groups=collections.Counter()
def group(name):
 name=re.sub(r'layers\.\d+','layers.*',name)
 return re.sub(r'experts\.\d+','experts.*',name)
for shard in sorted(set(ia.values())):
 ha,sa=read_header(a/shard);hb,sb=read_header(b/shard);hr,sr=read_header(base/shard)
 names=set(ha)-{'__metadata__'};assert names==set(hb)-{'__metadata__'}
 if os.path.samefile(a/shard,b/shard):same+=len(names);shards.append({'name':shard,'identical':True,'proof':'same_inode'});continue
 num=0
 with (a/shard).open('rb') as fa,(b/shard).open('rb') as fb,(base/shard).open('rb') as fr:
  for name in sorted(names):
   x=ha[name];y=hb[name];z=hr[name];assert x['shape']==y['shape'] and x['dtype']==y['dtype']
   length=x['data_offsets'][1]-x['data_offsets'][0];assert length==y['data_offsets'][1]-y['data_offsets'][0]
   fa.seek(sa+x['data_offsets'][0]);fb.seek(sb+y['data_offsets'][0]);fr.seek(sr+z['data_offsets'][0])
   h1=hashlib.sha256();h2=hashlib.sha256();metric=collections.defaultdict(float);left=length;different_bytes=0;values_changed=0
   while left:
    n=min(left,4<<20);aa=fa.read(n);bb=fb.read(n);rr=fr.read(n);assert len(aa)==len(bb)==len(rr)==n;left-=n;h1.update(aa);h2.update(bb)
    if aa!=bb:
     different_bytes+=int(np.count_nonzero(np.frombuffer(aa,dtype=np.uint8)!=np.frombuffer(bb,dtype=np.uint8)))
    if x['dtype']=='BF16':
     va=bf16_to_float(aa);vb=bf16_to_float(bb);vr=bf16_to_float(rr);da=va-vr;db=vb-vr
     metric['dealign_norm2']+=norm2(va);metric['difference_norm2']+=norm2(vb-va);metric['dealign_delta_norm2']+=norm2(da);metric['new_delta_norm2']+=norm2(db);metric['delta_dot']+=float(np.einsum('i,i->',da,db,dtype=np.float64));metric['max_abs_difference']=max(metric['max_abs_difference'],float(np.max(np.abs(vb-va))) if va.size else 0)
     values_changed+=int(np.count_nonzero(va!=vb))
   if h1.digest()==h2.digest():same+=1;continue
   num+=1;groups[group(name)]+=1
   row={'name':name,'shard':shard,'dtype':x['dtype'],'shape':x['shape'],'bytes':length,'different_bytes':different_bytes,'dealign_sha256':h1.hexdigest(),'new_sha256':h2.hexdigest()}
   if x['dtype']=='BF16':
    denom=metric['dealign_delta_norm2']*metric['new_delta_norm2'];row.update(values_changed=values_changed,relative_l2=math.sqrt(metric['difference_norm2']/max(metric['dealign_norm2'],1e-30)),delta_cosine=None if denom==0 else metric['delta_dot']/math.sqrt(denom),**metric)
   changed.append(row)
 shards.append({'name':shard,'identical':num==0,'changed_tensors':num,'proof':'full_tensor_sha256'})
 print('COMPARED',shard,num,flush=True)
metadata={}
for name in ['config.json','hf_quant_config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','preprocessor_config.json','processor_config.json','model.safetensors.index.json']:
 if not (a/name).exists() and not (b/name).exists():continue
 if not (a/name).exists() or not (b/name).exists():metadata[name]={'same':False,'missing_in':'dealign' if not(a/name).exists() else 'new'};continue
 x=(a/name).read_bytes();y=(b/name).read_bytes();row={'same':x==y}
 if name.endswith('.json') and x!=y:
  dx=json.loads(x);dy=json.loads(y);row['same_json']=dx==dy;row['different_top_level_keys']=[k for k in dx.keys()|dy.keys() if dx.get(k)!=dy.get(k)]
 metadata[name]=row
result={'status':'complete','seconds':time.time()-start,'identical_model':not changed and all(x['same'] for x in metadata.values()),'same_tensors':same,'changed_tensors':len(changed),'same_shards':sum(x['identical'] for x in shards),'different_shards':sum(not x['identical'] for x in shards),'groups':dict(groups),'metadata':metadata,'shards':shards,'changes':changed}
out.write_text(json.dumps(result,indent=2));print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ['changes','shards']}),flush=True)
