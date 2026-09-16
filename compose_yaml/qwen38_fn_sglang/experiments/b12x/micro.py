"""Real layer-0 TP-rank-0 weights, identical input/routing, two kernel paths."""
import json,time,contextlib,statistics
from pathlib import Path
import torch
from safetensors import safe_open
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from sglang.srt.layers.quantization.modelopt_quant import swizzle_blockscale
import adapter
model=Path('/hf/hub/models--dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4/snapshots/be794b990578ef3031eccf9f28e675a289a09ee9')
idx=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
fields={k:[] for k in ('w1','s1','g1','a1','w2','s2','g2','a2')}
with contextlib.ExitStack() as stack:
 files={f:stack.enter_context(safe_open(model/f,framework='pt')) for f in sorted(set(v for k,v in idx.items() if 'layers.0.mlp.experts.' in k))}
 for e in range(512):
  def get(proj,field):
   k=f'model.language_model.layers.0.mlp.experts.{e}.{proj}_proj.{field}';return files[idx[k]].get_tensor(k)
  assert torch.equal(get('up','weight_scale_2'),get('gate','weight_scale_2')),'Separate gate/up scales need dedicated handling'
  for key,value in {'w1':torch.cat([get('up','weight')[:320],get('gate','weight')[:320]]),'s1':torch.cat([get('up','weight_scale')[:320],get('gate','weight_scale')[:320]]),'g1':get('up','weight_scale_2').reshape(()),'a1':torch.maximum(get('up','input_scale'),get('gate','input_scale')).reshape(()),'w2':get('down','weight')[:,:160],'s2':get('down','weight_scale')[:,:20],'g2':get('down','weight_scale_2').reshape(()),'a2':get('down','input_scale').reshape(())}.items():fields[key].append(value)
f={k:torch.stack(v).cuda().contiguous() for k,v in fields.items()};del fields
w1,w2=f['w1'],f['w2'];s1=swizzle_blockscale(f['s1']);s2=swizzle_blockscale(f['s2']);a1=f['a1'].max().reciprocal();a2=f['a2'].max().reciprocal();alpha1=f['g1']/a1;alpha2=f['g2']/a2
owner=adapter.prepare(w1,s1,alpha1,a1,w2,s2,alpha2,a2)
print('Prepared real TP2 expert weights',w1.shape,w2.shape,flush=True)
rows=[]
for m in (1,4,16,128,1024):
 torch.manual_seed(900+m);x=torch.randn(m,2560,device='cuda',dtype=torch.bfloat16)*.2;ids=torch.rand(m,512,device='cuda').topk(10,dim=-1).indices.to(torch.int32);weights=torch.rand(m,10,device='cuda');weights/=weights.sum(-1,keepdim=True)
 output=torch.empty_like(x)
 def base():return cutlass_fused_moe(output=output,input=x,token_selected_experts=ids,token_final_scales=weights,fc1_expert_weights=w1.view(torch.long),fc2_expert_weights=w2.view(torch.long),output_dtype=torch.bfloat16,quant_scales=[a1,s1.view(torch.int32),alpha1,a2,s2.view(torch.int32),alpha2],tp_size=2,tp_rank=0,ep_size=1,ep_rank=0,tune_max_num_tokens=m,activation_type=ActivationType.Swiglu)[0]
 def candidate():return adapter.run(owner,x,ids,weights)
 ref=base().clone();got=candidate();torch.cuda.synchronize()
 diff=(ref.float()-got.float());rel=float(diff.norm()/ref.float().norm());cos=float(torch.nn.functional.cosine_similarity(ref.float().flatten(),got.float().flatten(),dim=0))
 row={'tokens':m,'relative_l2':rel,'cosine':cos,'max_abs':float(diff.abs().max()),'finite':bool(got.isfinite().all())}
 for name,fn in [('cutlass',base),('b12x',candidate)]:
  for _ in range(3):fn()
  torch.cuda.synchronize();times=[]
  for _ in range(7):
   begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
   for i in range(20):fn()
   end.record();end.synchronize();times.append(begin.elapsed_time(end)/20)
  row[name+'_ms']=statistics.median(times)
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for _ in range(20):fn()
  g.replay();torch.cuda.synchronize();times=[]
  for _ in range(7):
   begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();g.replay();end.record();end.synchronize();times.append(begin.elapsed_time(end)/20)
  row[name+'_graph_ms']=statistics.median(times)

 row['speedup']=row['cutlass_ms']/row['b12x_ms'];rows.append(row);print(json.dumps(row),flush=True)
 Path('/results/micro.json').write_text(json.dumps({'rows':rows,'peak_allocated':torch.cuda.max_memory_allocated()},indent=2))
