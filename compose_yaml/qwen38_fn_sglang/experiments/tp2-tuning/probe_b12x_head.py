import torch,json
from pathlib import Path
from b12x.gemm.bf16_gemv import mm
torch.manual_seed(317)
w=torch.randn(124160,2560,device='cuda',dtype=torch.bfloat16)*.01
xbase=torch.randn(8192,2560,device='cuda',dtype=torch.bfloat16)*.01
# Compile before the stream/allocator checks.
mm(xbase[:1],w,output_dtype=torch.bfloat16);torch.cuda.synchronize()
rows=[]
for offset in [0,1023,2047,4095,8191]:
 for nondefault in [False,True]:
  stream=torch.cuda.Stream() if nondefault else torch.cuda.current_stream()
  with torch.cuda.stream(stream):
   for repeat in range(4):
    poison=torch.full((1,124160),float('nan'),device='cuda',dtype=torch.bfloat16);del poison
    x=xbase[offset:offset+1];got=mm(x,w,output_dtype=torch.bfloat16).clone()
    ref=torch.nn.functional.linear(x,w)
    finite=bool(torch.isfinite(got).all());rel=float(torch.linalg.vector_norm(got.float()-ref.float())/torch.linalg.vector_norm(ref.float()))
    rows.append({'offset':offset,'nondefault_stream':nondefault,'repeat':repeat,'finite':finite,'relative_l2':rel})
  torch.cuda.synchronize()
print(json.dumps({'tests':len(rows),'nonfinite':sum(not r['finite'] for r in rows),'max_relative_l2':max(r['relative_l2'] for r in rows)}),flush=True)
Path('/results/b12x-head-probe.json').write_text(json.dumps(rows,indent=2))
