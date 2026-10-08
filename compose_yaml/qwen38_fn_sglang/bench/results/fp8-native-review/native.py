import json,re
import torch,triton,triton.language as tl
@triton.jit
def probe(OUT):
 lane=tl.arange(0,32)
 ones=tl.full((32,),0x38383838,tl.int32)
 zero=tl.full((32,),0,tl.float32)
 a,b,c,d=tl.inline_asm_elementwise(
  'mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {$0,$1,$2,$3}, {$4,$4,$4,$4}, {$4,$4}, {$5,$5,$5,$5};',
  constraints='=f,=f,=f,=f,r,f',args=[ones,zero],dtype=(tl.float32,tl.float32,tl.float32,tl.float32),is_pure=True,pack=1)
 tl.store(OUT+lane*4,a);tl.store(OUT+lane*4+1,b);tl.store(OUT+lane*4+2,c);tl.store(OUT+lane*4+3,d)
out=torch.empty(128,device='cuda')
k=probe[(1,)](out,num_warps=1);torch.cuda.synchronize()
assert torch.all(out==32).item(),out.tolist()
print(json.dumps({'capability':torch.cuda.get_device_capability(),'expected':32,'all_128_outputs_correct':True,'mma':sorted(set(re.findall(r'mma[.a-z0-9_]+',k.asm['ptx']))),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}))
