import os,torch,torch.distributed as dist
from tp_vocab import assemble_shortlist
dist.init_process_group('gloo');rank=dist.get_rank();world=dist.get_world_size()
full=torch.arange(32,dtype=torch.float32).reshape(8,4)
ids=torch.tensor([7,1,5,0,6,2])
def reduce(x):dist.all_reduce(x);return x
local=assemble_shortlist(full.chunk(2)[rank],ids,rank*4,rank*4+4,rank,world,reduce)
x=torch.tensor([[1.,2.,3.,4.],[-1.,2.,-3.,4.]])
y=x@local.T;parts=[torch.empty_like(y) for _ in range(world)];dist.all_gather(parts,y)
assert torch.equal(torch.cat(parts,dim=1),x@full[ids].T)
assert torch.equal(full,torch.arange(32,dtype=torch.float32).reshape(8,4))
if rank==0:print('TP2 shortlist: noncontiguous token mapping and gathered logits exact; original weights unchanged')
dist.destroy_process_group()
