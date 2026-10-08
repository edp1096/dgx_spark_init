"""Resident-weight CUDA-graph comparison for QAD BF16 target and draft heads."""
import json
import torch
import triton.testing
from b12x.gemm.bf16_gemv import mm

torch.manual_seed(381)
print(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}),flush=True)
for n in (65536,248320):
    w=torch.randn(n,2560,device='cuda',dtype=torch.bfloat16)*.02
    for m in (1,2,4,8):
        x=torch.randn(m,2560,device='cuda',dtype=torch.bfloat16)
        def native():return torch.nn.functional.linear(x,w)
        def b12x():return mm(x,w,output_dtype=torch.bfloat16)
        ref=native();out=b12x()
        error=float((out.float()-ref.float()).norm()/ref.float().norm())
        top1=bool(torch.equal(out.argmax(-1),ref.argmax(-1)))
        torch.testing.assert_close(out,ref,atol=.032,rtol=.032)
        timings={}
        for label,fn in [('torch',native),('b12x',b12x)]:
            for _ in range(3):fn()
            timings[label+'_us']=float(triton.testing.do_bench_cudagraph(fn,rep=150))*1000
        print(json.dumps(dict(n=n,m=m,relative_l2=error,top1_equal=top1,**timings)),flush=True)
    del w
