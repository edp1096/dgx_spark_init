import sys,json,pathlib
sys.path.insert(0,'/bridge')
import b12x_bridge as b
import torch
torch.set_num_threads(4)
handle=b.load('model.language_model.layers.0.mlp')
H=2560;top=10
torch.manual_seed(426)
row=torch.randn((1,H),device='cuda',dtype=torch.bfloat16)*.1
route=torch.rand((1,512),device='cuda').argsort(-1)[:,:top].int()
weight=torch.rand((1,top),device='cuda');weight/=weight.sum()
first=None;rows=[]
for R in [1,2,3,4,5,8,16]:
 x=row.repeat(R,1)
 ids=torch.full((R,top+1),512,device='cuda',dtype=torch.int32);ids[:,:top]=route
 w=torch.ones((R,top+1),device='cuda');w[:,:top]=weight
 saved=w.clone();y=torch.zeros((R,(top+1)*H),device='cuda')
 stream=torch.cuda.Stream()
 b.capture(handle,stream.cuda_stream,x.data_ptr(),ids.data_ptr(),w.data_ptr(),y.data_ptr(),R,H,top,4)
 graph=b.RETAIN[-1][0];graph.instantiate()
 with torch.cuda.stream(stream):w.copy_(saved);graph.replay()
 stream.synchronize()
 actual=y[0,:H].clone()
 if first is None:first=actual.clone()
 result={'rows':R,'equal_to_row1':torch.equal(actual,first),'max_abs':float((actual-first).abs().max()),'relative_l2':float((actual-first).norm()/first.norm())}
 rows.append(result);print(json.dumps(result),flush=True)
pathlib.Path('/bridge/width-check-result.json').write_text(json.dumps(rows,indent=2))
