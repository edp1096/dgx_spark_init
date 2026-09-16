import os,json,time,datetime
import torch
import torch.distributed as dist
rank=int(os.environ['RANK']);torch.cuda.set_device(0)
dist.init_process_group('nccl',rank=rank,world_size=2,timeout=datetime.timedelta(seconds=90))
for n in (4096,4194304):
    x=torch.empty(n,device='cuda',dtype=torch.float32)
    start=time.monotonic()
    for _ in range(10):
        x.fill_(rank+1);dist.all_reduce(x)
    torch.cuda.synchronize();assert torch.all(x==3).item()
    print(json.dumps({'rank':rank,'elements':n,'ms':(time.monotonic()-start)*100,'exact':True}),flush=True)
dist.barrier();dist.destroy_process_group()
