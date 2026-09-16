import torch,json
from pathlib import Path
from sglang.kernels.ops.elementwise.fast_topk import fast_topk
fail=[];count=0;torch.manual_seed(1918)
for length in [513,2048,6312,8192,50412]:
 for mode in ['random','ties','zero']:
  for repeat in range(10):
   values=torch.randn(4,length,device='cuda')
   if mode=='ties':values=values.round()
   if mode=='zero':values.zero_()
   score=torch.full((4,262144),float('nan'),device='cuda');score[:,:length]=values
   lengths=torch.full((4,),length,device='cuda',dtype=torch.int32)
   actual=fast_topk(score,lengths,512)
   valid=bool(((actual>=0)&(actual<length)).all())
   good=valid
   if valid:
    chosen=values.gather(1,actual.long());threshold=values.topk(512,dim=1).values[:,-1:]
    ordered=actual.sort(dim=1).values
    good=bool((chosen>=threshold).all() and (ordered[:,1:]!=ordered[:,:-1]).all())
   if not good:fail.append({'length':length,'mode':mode,'repeat':repeat,'valid_indices':valid})
   count+=1
   torch.cuda.synchronize()
result={'tests':count,'failures':fail};print(json.dumps(result),flush=True);Path('/results/fast-topk-edge-probe.json').write_text(json.dumps(result,indent=2))
