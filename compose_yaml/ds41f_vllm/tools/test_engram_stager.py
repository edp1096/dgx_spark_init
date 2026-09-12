"""Compare real TP-sharded staged rows with the existing eager disk lookup."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json,sys,types
from pathlib import Path
import torch
from vllm.models.deepseek_v4_1.common.engram import (
    Engram,EngramLayout,NgramHashState,DiskEngramTable,ParallelEngramEmbedding,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import image_sentinel_mask
import engram_stager

model=sys.argv[1]
cfg=types.SimpleNamespace(**json.loads((Path(model)/'config.json').read_text())['text_config'])
layout=EngramLayout(cfg)
configuration=types.SimpleNamespace(
    use_v2_model_runner=True,
    model_config=types.SimpleNamespace(tokenizer=model,trust_remote_code=False,revision=None),
    scheduler_config=types.SimpleNamespace(max_num_batched_tokens=16),
    parallel_config=types.SimpleNamespace(data_parallel_size=1,pipeline_parallel_size=1,num_ubatches=0,enable_dbo=False))
with torch.device('cuda'):
    state=NgramHashState(configuration,layout,
        types.SimpleNamespace(block_size=128,kv_cache=torch.zeros(1)))
state.to('cuda')
engram_stager.get_engram_dp_size=lambda:1
for rank in (0,1):
    owner=torch.nn.Module();owner.engram_hash=state
    owner.layers=torch.nn.ModuleList()
    width=(layout.n_hash_cols+1)//2;start=rank*width
    for index,layer in enumerate(layout.layer_ids):
        engram=Engram.__new__(Engram);torch.nn.Module.__init__(engram)
        engram.layer_hash_index=index
        offsets=layout.offsets[index].tolist()+[layout.num_embeddings[index]]
        emb=types.SimpleNamespace(part_n_hash_cols=width,head_start=start,
            n_hash_cols=layout.n_hash_cols,dim=layout.head_dim,
            vocab_start_idx=offsets[start],vocab_end_idx=offsets[min(start+width,layout.n_hash_cols)])
        emb.disk=DiskEngramTable(model,layer,emb.dim,32,emb.vocab_start_idx,
                                emb.vocab_end_idx-emb.vocab_start_idx)
        engram.embed_tokens=emb
        engram.staged_rows=torch.empty(16,width,emb.dim,device='cuda',dtype=torch.bfloat16)
        owner.layers.append(engram)
    stager=engram_stager.EngramDiskStager(configuration,owner)
    addresses=[e.staged_rows.data_ptr() for e in owner.layers]
    for n,position in ((1,0),(6,7),(3,12),(6,15)):
        ids=torch.arange(100,100+n,device='cuda',dtype=torch.int32)
        pos=torch.arange(position,position+n,device='cuda',dtype=torch.int64)
        query=torch.tensor([0,n],device='cuda',dtype=torch.int32)
        lookback=torch.full((1,state.lookback_depth),-1 if position==0 else 77,
                            device='cuda',dtype=torch.int32)
        stager.stage(ids,pos,query,lookback,n)
        hashes=state(ids,pos,query,image_sentinel_mask(ids),lookback,
                     image_sentinel_mask(lookback),None,None)
        torch.testing.assert_close(stager.hashes[:n],hashes,rtol=0,atol=0)
        for engram in owner.layers:
            expected=torch.empty_like(engram.staged_rows[:n])
            ParallelEngramEmbedding.lookup(engram.embed_tokens,
                hashes[:,engram.layer_hash_index],expected)
            torch.testing.assert_close(engram.staged_rows[:n],expected,rtol=0,atol=0)
        assert addresses==[e.staged_rows.data_ptr() for e in owner.layers]
    stager.dummy(6)
    assert not stager.keep[:6].any().item()
    print('ENGRAM_PRESTAGE_EXACT_PASS rank',rank,flush=True)
