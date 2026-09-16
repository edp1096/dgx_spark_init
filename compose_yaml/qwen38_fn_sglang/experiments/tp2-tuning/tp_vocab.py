"""Build balanced TP shards of a global MTP shortlist once at startup.

Main-model weights and its full-vocabulary verification remain unchanged.
"""
import torch

def assemble_shortlist(weight, ids, start, end, rank, world, reduce):
    assert weight.ndim==2 and ids.ndim==1
    assert ids.numel()%world==0 and ids.unique().numel()==ids.numel()
    assert end-start<=weight.shape[0]
    selected=torch.zeros((ids.numel(),weight.shape[1]),dtype=weight.dtype,device=weight.device)
    owned=(ids>=start)&(ids<end)
    positions=owned.nonzero().flatten()
    selected[positions]=weight.index_select(0,ids[positions]-start)
    selected=reduce(selected)
    # All-gather of output logits must follow shortlist order, not original
    # token ID ranges. Balance by shortlist position to avoid an idle TP rank.
    return selected.chunk(world,dim=0)[rank].contiguous().clone()

def make_tp_shortlist(head, target_lm_head, ids):
    from sglang.srt.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce
    rank=get_tensor_model_parallel_rank();world=get_tensor_model_parallel_world_size()
    assert world==2, 'Only TP2 has been qualified by this adapter'
    shard=target_lm_head.shard_indices
    assert int(ids.min())>=0 and int(ids.max())<target_lm_head.org_vocab_size
    local=assemble_shortlist(head,ids,int(shard.org_vocab_start_index),int(shard.org_vocab_end_index),rank,world,tensor_model_parallel_all_reduce)
    print(f'QWEN_TP_SHORTLIST rank={rank} world={world} global={len(ids)} local={len(local)}',flush=True)
    return torch.nn.Parameter(local,requires_grad=False)
