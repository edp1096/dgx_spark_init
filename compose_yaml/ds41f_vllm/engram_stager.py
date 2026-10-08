"""Exact Engram disk reads before model execution, for this TP2/V2 runtime.

Adapted from the upstream EngramDiskStager to the pinned DP-aware interface.
DP sharing, pipeline parallelism and microbatch overlap are deliberately
rejected here; the qualified configuration has one TP2 request at a time.
"""
import torch
from vllm.models.deepseek_v4_1.common.engram import (
    Engram,NgramHashState,gather_dequant_many,get_engram_dp_size,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import image_sentinel_mask

class EngramDiskStager:
    def __init__(self,config,model):
        parallel=config.parallel_config
        if (parallel.data_parallel_size!=1 or parallel.pipeline_parallel_size!=1
                or parallel.enable_dbo or parallel.num_ubatches not in (0,1)
                or get_engram_dp_size()!=1):
            raise ValueError('Engram prestaging requires DP1/PP1 and one microbatch')
        hashes=[m for m in model.modules() if isinstance(m,NgramHashState)]
        self.engrams=sorted([m for m in model.modules() if isinstance(m,Engram)],
                            key=lambda m:m.layer_hash_index)
        if len(hashes)!=1 or not self.engrams:
            raise ValueError('Expected one n-gram state and at least one Engram layer')
        self.hash_state=hashes[0]
        if self.hash_state.use_slot_cache:
            raise ValueError('Engram prestaging requires the V2 lookback-window path')
        self.max_tokens=config.scheduler_config.max_num_batched_tokens
        device=self.hash_state.token_map.device
        layers=self.hash_state.multipliers.shape[0]
        columns=self.hash_state.layout.n_hash_cols
        self.hashes=torch.zeros(self.max_tokens,layers,columns,dtype=torch.int32,device=device)
        self.keep=torch.zeros(self.max_tokens,dtype=torch.bool,device=device)
        emb=self.engrams[0].embed_tokens
        self.head_start=emb.head_start
        self.head_end=min(emb.head_start+emb.part_n_hash_cols,emb.n_hash_cols)
        self.host_hashes=torch.empty(self.max_tokens,layers,self.head_end-self.head_start,
                                    dtype=torch.int32,pin_memory=True,device='cpu')
        self.host_rows=[]
        for engram in self.engrams:
            if engram.embed_tokens.disk is None: raise ValueError('Expected disk Engram')
            engram.staged_rows.zero_()
            engram.disk_prestaged=True
            self.host_rows.append(torch.empty_like(engram.staged_rows,device='cpu',pin_memory=True))
        self.ready=torch.cuda.Event()
        owners=[m for m in model.modules() if m._modules.get('engram_hash') is self.hash_state]
        if len(owners)!=1: raise ValueError('Expected a unique Engram model owner')
        owners[0].disk_stager=self
        self.stages=0
        print(f'ENGRAM_PRESTAGE enabled: {len(self.engrams)} tables, heads '
              f'[{self.head_start},{self.head_end}), capacity={self.max_tokens}',flush=True)

    def dummy(self,n):
        self.hashes[:n].zero_();self.keep[:n].zero_()
        for engram in self.engrams: engram.staged_rows[:n].zero_()

    def stage(self,input_ids,positions,query_start_loc,lookback,n):
        n=int(n)
        if not 0<=n<=self.max_tokens: raise ValueError('Invalid Engram batch size')
        if n==0 or not self.hash_state.ensure_cache():
            self.dummy(n)
            return
        ids=input_ids[:n];dead=image_sentinel_mask(ids)
        hashes=self.hash_state(ids,positions[:n],query_start_loc,dead,
                               lookback,image_sentinel_mask(lookback),None,None)
        self.hashes[:n].copy_(hashes)
        self.keep[:n].copy_(~dead)
        host=self.host_hashes[:n]
        host.copy_(hashes[:,:,self.head_start:self.head_end],non_blocking=True)
        self.ready.record();self.ready.synchronize()
        requests=[]
        for engram in self.engrams:
            emb=engram.embed_tokens
            local=host[:,engram.layer_hash_index].to(torch.int64)
            if local.shape[1]<emb.part_n_hash_cols:
                local=torch.cat((local,torch.full((n,emb.part_n_hash_cols-local.shape[1]),
                                -1,dtype=torch.int64,device='cpu')),dim=1)
            rows=local.reshape(-1)
            owned=(rows>=emb.vocab_start_idx)&(rows<emb.vocab_end_idx)
            rel=torch.where(owned,rows-emb.vocab_start_idx,torch.zeros_like(rows))
            requests.append((emb.disk,rel,owned))
        for engram,rows,buffer in zip(self.engrams,gather_dequant_many(requests),self.host_rows):
            staged=buffer[:n]
            staged.copy_(rows.reshape(staged.shape))
            engram.staged_rows[:n].copy_(staged,non_blocking=True)
        self.stages+=1
