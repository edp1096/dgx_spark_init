"""Compare packed selected-KV attention, preserving the QSA selector and BF16."""
import json,statistics,torch
from pathlib import Path
from sglang.kernels.ops.attention import qwen38_qsa_sm121_varlen as baseline
from b12x.attention import varlen as b
rows=[]
for batch in (1,4,16,128):
 torch.manual_seed(batch);length=2051
 q=torch.randn(batch,12,256,device='cuda',dtype=torch.bfloat16)
 k=torch.randn(batch*length,1,256,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
 cq=torch.arange(batch+1,device='cuda',dtype=torch.int32);ck=cq*length
 def old():return baseline(q,k,v,cq,ck,max_seqlen_q=1,max_seqlen_k=length,softmax_scale=1/16,causal=True)
 row={'queries':batch,'selected_keys':length}
 try:
  plan=b.create_plan(q,k,v,cq,ck,max_seqlen_q=1,max_seqlen_k=length,causal=True);sp=b.plan(plan);scratch=tuple(torch.empty(x.shape,dtype=x.dtype,device=x.device) for x in sp.scratch_specs());binding=sp.bind(scratch=scratch,q=q,k=k,v=v,cu_seqlens_q=cq,cu_seqlens_k=ck,max_seqlen_q=1,max_seqlen_k=length,softmax_scale=1/16,causal=True)
  def new():return b.run(binding=binding)[0]
  ref=old();got=new();torch.cuda.synchronize();row['relative_l2']=float((ref.float()-got.float()).norm()/ref.float().norm());row['max_abs']=float((ref.float()-got.float()).abs().max())
  for name,fn in [('sglang',old),('b12x',new)]:
   for _ in range(3):fn()
   torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):
    for _ in range(20):fn()
   graph.replay();torch.cuda.synchronize();times=[]
   for _ in range(7):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();graph.replay();end.record();end.synchronize();times.append(start.elapsed_time(end)/20)
   row[name+'_graph_ms']=statistics.median(times)
 except Exception as e:row['error']=repr(e)
 rows.append(row);print(json.dumps(row),flush=True);Path('/results/qsa-micro.json').write_text(json.dumps({'rows':rows},indent=2))
