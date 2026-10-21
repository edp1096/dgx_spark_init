import sys,os,json,pathlib
sys.path.insert(0,'/bridge')
import b12x_bridge as b
import torch
torch.set_num_threads(4);handle=b.load('model.language_model.layers.0.mlp');H=2560;top=10;results=[]
def execute(x,ids,w):
 R=x.shape[0];y=torch.zeros((R,(top+1)*H),device='cuda');saved=w.clone();stream=torch.cuda.Stream()
 b.capture(handle,stream.cuda_stream,x.data_ptr(),ids.data_ptr(),w.data_ptr(),y.data_ptr(),R,H,top,4)
 graph=b.RETAIN[-1][0];graph.instantiate()
 with torch.cuda.stream(stream):w.copy_(saved);graph.replay()
 stream.synchronize();first=y[:,:H].clone()
 with torch.cuda.stream(stream):w.copy_(saved);graph.replay()
 stream.synchronize();assert torch.equal(first,y[:,:H]),'Replay is not deterministic'
 return first
for amplitude in (1.,):
 torch.manual_seed(714);token=torch.randn((1,H),device='cuda',dtype=torch.bfloat16)*amplitude;chosen=torch.rand((1,512),device='cuda').argsort(-1)[:,:top].int();weight=torch.rand((1,top),device='cuda');weight/=weight.sum()
 ids=torch.full((1,top+1),512,device='cuda',dtype=torch.int32);ids[:,:top]=chosen;w=torch.ones((1,top+1),device='cuda');w[:,:top]=weight
 reference=execute(token,ids,w)[0]
 for R in (2048,):
  for collide in (False,):
   torch.manual_seed(900+R);x=torch.randn((R,H),device='cuda',dtype=torch.bfloat16)*amplitude;x[-1]=token[0]
   ids=torch.full((R,top+1),512,device='cuda',dtype=torch.int32);ids[:,:top]=chosen if collide else torch.rand((R,512),device='cuda').argsort(-1)[:,:top].int();ids[-1,:top]=chosen[0]
   w=torch.ones((R,top+1),device='cuda');w[:,:top]=torch.rand((R,top),device='cuda');w[:,:top]/=w[:,:top].sum(-1,keepdim=True);w[-1,:top]=weight[0]
   actual=execute(x,ids,w)[-1];delta=actual-reference
   result={'rows':R,'position':R-1,'amplitude':amplitude,'all_rows_route_to_same_experts':collide,'equal':torch.equal(actual,reference),'max_abs':float(delta.abs().max()),'relative_l2':float(delta.norm()/reference.norm())};results.append(result);print(result,flush=True)
   pathlib.Path('/bridge/'+sys.argv[1]+'.json').write_text(json.dumps(results,indent=2))
