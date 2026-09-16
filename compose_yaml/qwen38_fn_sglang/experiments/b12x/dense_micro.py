"""Representative TP2 dense geometries, original BF16/FP32 precision."""
import json,statistics,torch
from pathlib import Path
from b12x.gemm.bf16_gemv import mm
rows=[]
for label,n,k,dtype in [('gate_proj',48,2560,torch.bfloat16),('qkvz',8192,2560,torch.bfloat16),('output_proj',2560,3072,torch.bfloat16),('lm_head',124160,2560,torch.bfloat16),('router_fp32',512,2560,torch.float32)]:
 torch.manual_seed(984);weight=torch.randn(n,k,device='cuda',dtype=dtype)*.02
 for m in (1,4,128):
  row={'geometry':label,'m':m,'n':n,'k':k,'dtype':str(dtype)}
  try:
   x=torch.randn(m,k,device='cuda',dtype=dtype)*.1;out=torch.empty(m,n,device='cuda',dtype=dtype)
   def old():return torch.mm(x,weight.T)
   def new():return mm(x,weight,out=out)
   ref=old().clone();got=new().clone();torch.cuda.synchronize();row['relative_l2']=float((ref.float()-got.float()).norm()/ref.float().norm());row['max_abs']=float((ref.float()-got.float()).abs().max());row['finite']=bool(got.isfinite().all())
   for name,fn in [('torch',old),('b12x',new)]:
    for _ in range(3):fn()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     for _ in range(10):fn()
    graph.replay();torch.cuda.synchronize();times=[]
    for _ in range(7):
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();graph.replay();end.record();end.synchronize();times.append(start.elapsed_time(end)/10)
    row[name+'_graph_ms']=statistics.median(times)
  except Exception as e:row['error']=repr(e)
  rows.append(row);print(json.dumps(row),flush=True);Path('/results/dense-micro.json').write_text(json.dumps({'rows':rows},indent=2))
 del weight
 torch.cuda.empty_cache()
