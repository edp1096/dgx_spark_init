import sys,json,pathlib
sys.path.insert(0,'/bridge')
import b12x_bridge as b
import torch
torch.set_num_threads(4);handle=b.load('model.language_model.layers.0.mlp');H=2560;top=10
stream=torch.cuda.Stream();buffers={}
for R in [1,2,3,4,5]:
 x=torch.randn((R,H),device='cuda',dtype=torch.bfloat16)
 ids=torch.zeros((R,11),device='cuda',dtype=torch.int32)
 w=torch.ones((R,11),device='cuda')/10
 y=torch.zeros((R,11*H),device='cuda')
 b.capture(handle,stream.cuda_stream,x.data_ptr(),ids.data_ptr(),w.data_ptr(),y.data_ptr(),R,H,top,4)
 graph=b.RETAIN[-1][0];graph.instantiate();buffers[R]=(x,ids,w,y,graph)
rows=[]
for seed in range(50):
 torch.manual_seed(800+seed)
 x=torch.randn((5,H),device='cuda',dtype=torch.bfloat16)
 ids=torch.full((5,11),512,device='cuda',dtype=torch.int32);ids[:,:10]=torch.rand((5,512),device='cuda').argsort(-1)[:,:10].int()
 w=torch.ones((5,11),device='cuda');w[:,:10]=torch.rand((5,10),device='cuda');w[:,:10]/=w[:,:10].sum(-1,keepdim=True)
 torch.cuda.synchronize();outputs=[]
 for R,(a,p,weights,y,g) in buffers.items():
  with torch.cuda.stream(stream):a.copy_(x[:R]);p.copy_(ids[:R]);weights.copy_(w[:R]);g.replay()
  stream.synchronize();outputs.append(y[0,:H].clone())
 torch.cuda.synchronize()
 row={'seed':seed,'equal':[torch.equal(outputs[0],v) for v in outputs],'max_abs':[float((outputs[0]-v).abs().max()) for v in outputs]};rows.append(row)
 if not all(row['equal']):print(json.dumps(row),flush=True)
pathlib.Path('/bridge/width-stress-result.json').write_text(json.dumps(rows,indent=2));print(json.dumps({'cases':50,'passing':sum(all(x['equal']) for x in rows)}),flush=True)
