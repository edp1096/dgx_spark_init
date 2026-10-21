import sys,pathlib,json,os,time,dataclasses
sys.path.insert(0,'/bridge')
import b12x_bridge as b
import torch
torch.set_num_threads(4)
handle=b.load('model.language_model.layers.0.mlp');H=2560;top=10
torch.manual_seed(426);fixed_x=torch.randn((1,H),device='cuda',dtype=torch.bfloat16)*.1;fixed_ids=torch.rand((1,512),device='cuda').argsort(-1)[:,:top].int();fixed_w=torch.rand((1,top),device='cuda');fixed_w/=fixed_w.sum()
rows=[];first=None
for R in [int(v) for v in os.environ.get('MICRO_ROWS','1,2,5,16,128').split(',')]:
 torch.manual_seed(426)
 x=torch.randn((R,H),device='cuda',dtype=torch.bfloat16)*.1
 ids=torch.full((R,top+1),512,device='cuda',dtype=torch.int32);ids[:,:top]=torch.rand((R,512),device='cuda').argsort(-1)[:,:top].int()
 w=torch.ones((R,top+1),device='cuda');w[:,:top]=torch.rand((R,top),device='cuda');w[:,:top]/=w[:,:top].sum(-1,keepdim=True)
 x[0]=fixed_x[0];ids[0,:top]=fixed_ids[0];w[0,:top]=fixed_w[0]
 y=torch.zeros((R,(top+1)*H),device='cuda');saved=w.clone();stream=torch.cuda.Stream()
 b.capture(handle,stream.cuda_stream,x.data_ptr(),ids.data_ptr(),w.data_ptr(),y.data_ptr(),R,H,top,4)
 graph=b.RETAIN[-1][0];graph.instantiate();binding=b.RETAIN[-1][1]
 def run():w.copy_(saved);graph.replay()
 with torch.cuda.stream(stream):run()
 stream.synchronize();expected=y.clone();differences=[]
 for j in range(5):
  with torch.cuda.stream(stream):run()
  stream.synchronize();differences.append(float((y-expected).abs().max()))
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 with torch.cuda.stream(stream):
  start.record()
  for j in range(50):run()
  end.record()
 stream.synchronize()
 p,scratch=b.PLANS[(R,top)]
 if first is None:first=y[0,:H].clone()
 same_width=torch.equal(y[0,:H],first);width_diff=float((y[0,:H]-first).abs().max())
 row={'width_equal':same_width,'width_max_abs':width_diff,'rows':R,'layout':os.getenv('TF_B12X_LAYOUT','source_native'),'fp32_router':os.getenv('B12X_W4A16_FP32_TOPK_WEIGHTS','0'),'total_ms':start.elapsed_time(end)/50,'replay_max_abs':max(differences),'repeat_equal':max(differences)==0,'binding_fields':list(vars(binding)),'plan_type':type(p).__name__,'scratch':list(scratch),'decode_config':str(binding.decode_config),'fused_launch':str(binding.fused_launch)}
 with torch.cuda.stream(stream):
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:run();stream.synchronize()
 row['cuda_profile']=prof.key_averages().table(sort_by='self_device_time_total',row_limit=12)
 rows.append(row);pathlib.Path('/bridge/'+os.environ.get('TRIAL_LABEL','micro-'+os.getenv('TF_B12X_LAYOUT','source_native')+'-fp'+os.getenv('B12X_W4A16_FP32_TOPK_WEIGHTS','0'))+'.json').write_text(json.dumps(rows,indent=2));print({k:v for k,v in row.items() if k not in ('cuda_profile','binding_fields','scratch','fused_launch')},flush=True)
