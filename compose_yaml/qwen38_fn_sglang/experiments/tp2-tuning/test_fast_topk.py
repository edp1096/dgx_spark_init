import torch,json,time
from pathlib import Path
from sglang.kernels.ops.elementwise.fast_topk import fast_topk
rows=[];torch.manual_seed(918)
for length in [511,512,513,1024,2048,4096,6312,8192,16384]:
 for repeat in range(20):
  batch=16;start=torch.arange(batch,device='cuda',dtype=torch.int32)*3
  lengths=torch.full((batch,),length,device='cuda',dtype=torch.int32)
  score=torch.full((batch,length+64),float('nan'),device='cuda',dtype=torch.float32)
  values=torch.randn(batch,length,device='cuda')
  for i in range(batch):score[i,int(start[i]):int(start[i])+length]=values[i]
  actual=fast_topk(score,lengths,512,row_starts=start)
  ref=torch.topk(values,min(length,512),dim=1).indices.sort(dim=1).values
  if length<512:actual=actual[:,:length]
  good=torch.equal(actual.sort(dim=1).values,ref)
  if not good:rows.append({'length':length,'repeat':repeat,'matching':False,'invalid':bool(((actual<0)|(actual>=length)).any())})
  torch.cuda.synchronize()
print(json.dumps({'tests':180,'failures':rows}),flush=True)
Path('/results/fast-topk-probe.json').write_text(json.dumps({'tests':180,'failures':rows},indent=2))
