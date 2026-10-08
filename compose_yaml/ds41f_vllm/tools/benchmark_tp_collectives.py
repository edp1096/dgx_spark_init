"""Compare this vLLM's PyNccl with b12x RoCEnante on real TP2 FP32 buffers."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json,os,statistics,time
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from b12x.comm import roce
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=100)
args=p.parse_args()
dist.init_process_group('gloo',timeout=timedelta(seconds=120));rank=dist.get_rank();world=dist.get_world_size()
assert world==2
device=torch.device('cuda',0);torch.cuda.set_device(device)
nccl=PyNcclCommunicator(group=dist.group.WORLD,device=device)
transport=roce.AllReduce.from_exchange_group(exchange_group=dist.group.WORLD,device=device,max_size=2<<20,max_gather_bytes=2<<20)
transport.prepare((torch.float32,torch.bfloat16))
dist.barrier()
rows=[]

def timing(fn):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();fn();end.record();end.synchronize();transport.check_health()
    return start.elapsed_time(end)*1000

try:
    for dtype in (torch.float32,torch.bfloat16):
        for nbytes in (20480,122880,262144):
            n=nbytes//torch.tensor([],dtype=dtype).element_size()
            torch.manual_seed(938+rank);x=torch.randn(n,device=device,dtype=dtype)
            a=torch.empty_like(x);b=torch.empty_like(x)
            def native():nccl.all_reduce(x,a)
            def direct():transport.all_reduce(x,out=b)
            native();direct();torch.cuda.synchronize();transport.check_health()
            # With two ranks each element has exactly one addition.
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            for _ in range(10):native();direct()
            torch.cuda.synchronize()
            samples={'pynccl':[],'roce':[]}
            for block in range(4):
                arms=(('pynccl',native),('roce',direct))
                if block%2:arms=arms[::-1]
                for name,fn in arms:
                    dist.barrier()
                    samples[name].extend(timing(fn) for _ in range(args.samples//4))
            native();direct();torch.cuda.synchronize();transport.check_health()
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            gathered=[None]*world;dist.all_gather_object(gathered,samples)
            if rank==0:
                result={'operation':'all_reduce','dtype':str(dtype),'bytes':nbytes,'exact':True,
                        'slowest_rank_median_us':{k:max(statistics.median(s[k]) for s in gathered) for k in samples},'samples_by_rank':gathered}
                rows.append(result);print(json.dumps({k:v for k,v in result.items() if k!='samples_by_rank'}),flush=True)
        # dim-0 buffers measure the transport; the serving test also checks
        # last-dimension gathering and its existing reshape/copy overhead.
        x=torch.randn(6,64512,device=device,dtype=dtype)
        a=torch.empty(12,64512,device=device,dtype=dtype)
        def native_gather():nccl.all_gather(a,x)
        def direct_gather():return transport.all_gather(x,dim=0)
        native_gather();b=direct_gather();torch.cuda.synchronize();transport.check_health()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        samples={'pynccl':[],'roce':[]}
        for block in range(4):
            arms=(('pynccl',native_gather),('roce',direct_gather))
            if block%2:arms=arms[::-1]
            for name,fn in arms:
                for _ in range(10):fn()
                torch.cuda.synchronize();dist.barrier()
                samples[name].extend(timing(fn) for _ in range(args.samples//4))
        gathered=[None]*world;dist.all_gather_object(gathered,samples)
        if rank==0:
            result={'operation':'all_gather','dtype':str(dtype),'bytes':x.numel()*x.element_size(),'exact':True,
                    'slowest_rank_median_us':{k:max(statistics.median(s[k]) for s in gathered) for k in samples},'samples_by_rank':gathered}
            rows.append(result);print(json.dumps({k:v for k,v in result.items() if k!='samples_by_rank'}),flush=True)
    if rank==0:
        Path(args.output).write_text(json.dumps({'b12x_revision':'789bbb3c846565c41f3404af3e0d7c9ce8702f7f','world':world,'torch':torch.__version__,'rows':rows},indent=2))
finally:
    torch.cuda.synchronize();transport.close();nccl.destroy();dist.destroy_process_group()
