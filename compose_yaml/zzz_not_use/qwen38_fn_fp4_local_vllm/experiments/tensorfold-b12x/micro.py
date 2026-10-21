import sys,pathlib,time,json,ctypes
sys.path.insert(0,"/bridge")
import b12x_bridge as b
import torch
b.torch.set_num_threads(4)
id=b.load("model.language_model.layers.0.mlp")
H=2560;N=640;top=10
ex,wp,*_=b.LAYERS[id]
results=[]
def dequant(base):
 w=b.raw(base+".weight").cuda();s=b.scales(base,w.shape[0],w.shape[1]*2).cuda().view(torch.float8_e4m3fn).float();g=b.raw(base+".weight_scale_2").cuda()
 codes=torch.stack((w&15,w>>4),dim=-1).reshape(w.shape[0],-1).long()
 lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6],device="cuda")
 return lut[codes]*s.repeat_interleave(16,dim=1)*g
for R in [1,5,16,128]:
 torch.manual_seed(1200+R)
 x=torch.randn((R,H),device="cuda",dtype=torch.bfloat16)*.1
 picks=torch.empty((R,top+1),device="cuda",dtype=torch.int32);picks[:,:top]=torch.rand((R,512),device="cuda").argsort(-1)[:,:top].int();picks[:,-1]=512
 wts=torch.rand((R,top+1),device="cuda");wts[:,:top]/=wts[:,:top].sum(-1,keepdim=True)
 y=torch.zeros((R,(top+1)*H),device="cuda",dtype=torch.float32);y[:,top*H:]=123.25;weights=wts.clone();input_before=x.clone();ids_before=picks.clone()
 stream=torch.cuda.Stream()
 ptr=b.capture(id,stream.cuda_stream,x.data_ptr(),picks.data_ptr(),wts.data_ptr(),y.data_ptr(),R,H,top,4)
 graph=b.RETAIN[-1][0];graph.instantiate()
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 # Every replay must restore the router's input weights before the adapter rewrites them.
 with torch.cuda.stream(stream):
  start.record()
  for i in range(20):wts.copy_(weights);graph.replay()
  end.record()
 stream.synchronize();ms=start.elapsed_time(end)/20
 assert torch.isfinite(y).all()
 assert torch.equal(x,input_before) and torch.equal(picks,ids_before)
 assert torch.equal(y[:,top*H:],torch.full_like(y[:,top*H:],123.25))
 expected=y.clone();torch.cuda.synchronize()
 for _ in range(5):
  with torch.cuda.stream(stream):wts.copy_(weights);graph.replay()
  stream.synchronize();assert torch.equal(y,expected), "Repeat changed output bytes"
 # Validate selected experts against their original checkpoint values for one input row.
 ref=torch.zeros(H,device="cuda")
 for j in range(top):
  e=int(picks[0,j]);base=f"model.language_model.layers.0.mlp.experts.{e}"
  gate=dequant(base+".gate_proj")@x[0].float();up=dequant(base+".up_proj")@x[0].float()
  act=(torch.nn.functional.silu(gate)*up).bfloat16().float()
  out=dequant(base+".down_proj")@act
  ref+=out*weights[0,j]
 actual=y[0,:H];rel=float((actual-ref).norm()/ref.norm());cos=float(torch.nn.functional.cosine_similarity(actual,ref,dim=0))
 row={"rows":R,"b12x_graph_ms":ms,"relative_l2_vs_fp32_reference":rel,"cosine":cos};results.append(row);print(json.dumps(row),flush=True)
 assert rel<.03 and cos>.999, row
pathlib.Path("/bridge/micro-result.json").write_text(json.dumps(results,indent=2))
