import json,re
import torch,triton,triton.language as tl
@triton.jit
def dot(A,B,C,K:tl.constexpr,ACC:tl.constexpr):
 m=tl.arange(0,16);n=tl.arange(0,64);k=tl.arange(0,K)
 a=tl.load(A+m[:,None]*K+k[None,:]);b=tl.load(B+k[:,None]*64+n[None,:])
 c=tl.dot(a,b,max_num_imprecise_acc=ACC)
 tl.store(C+m[:,None]*64+n[None,:],c)
torch.manual_seed(42)
a=(torch.randn(16,256,device='cuda')*.2).to(torch.float8_e4m3fn)
b=(torch.randn(256,64,device='cuda')*.2).to(torch.float8_e4m3fn)
c=torch.empty(16,64,device='cuda',dtype=torch.float32)
for acc in (0,32,256):
 kernel=dot[(1,)](a,b,c,256,acc,num_warps=4)
 torch.cuda.synchronize();ref=a.float()@b.float();err=(c-ref).abs().max().item()
 ops=sorted(set(re.findall(r'mma[.a-z0-9_]+',kernel.asm['ptx'])))
 r={'capability':torch.cuda.get_device_capability(),'imprecise_acc':acc,'max_abs_error_vs_quantized_input_fp32':err,'mma_instructions':ops,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
 print(json.dumps(r),flush=True)
