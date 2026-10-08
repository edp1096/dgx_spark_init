"""One bounded, disposable next-prompt-chunk read; no expert prediction or math changes."""
import os
import time
from concurrent.futures import ThreadPoolExecutor


def next_chunk(batch, states, capacity, excluded=()):
    candidates=[]
    for i, req_id in enumerate(batch.req_ids):
        if req_id in excluded or not batch.is_prefilling_np[i]:
            continue
        index=int(batch.idx_mapping_np[i])
        end=int(batch.num_computed_prefill_tokens_np[i])+int(batch.num_scheduled_tokens[i])
        remaining=int(states.prompt_len.np[index])-end
        if remaining > 0:
            candidates.append((remaining,index,end))
    if not candidates:
        return None
    remaining,index,start=max(candidates)
    return index,start,min(remaining,capacity)


class Prefetcher:
    def __init__(self, config, model):
        from engram_reader import Reader
        from types import SimpleNamespace
        from vllm.models.deepseek_v4_1.common.engram import Engram,NgramHashState,get_engram_dp_size
        parallel=config.parallel_config
        if (get_engram_dp_size()!=1 or parallel.pipeline_parallel_size!=1
                or parallel.enable_dbo or parallel.num_ubatches not in (0,1)):
            raise ValueError('Next-chunk prefetch requires DP1/PP1')
        hashes=[m for m in model.modules() if isinstance(m,NgramHashState)]
        engrams=sorted([m for m in model.modules() if isinstance(m,Engram)],key=lambda m:m.layer_hash_index)
        if len(hashes)!=1 or hashes[0].use_slot_cache or not engrams:
            raise ValueError('Next-chunk prefetch requires stateless V2 hashes')
        if any(m.embed_tokens.disk is None for m in engrams):
            raise ValueError('Expected disk Engram tables')
        emb=engrams[0].embed_tokens
        self.stager=SimpleNamespace(hash_state=hashes[0],engrams=engrams,
            max_tokens=config.scheduler_config.max_num_batched_tokens,
            head_start=emb.head_start,head_end=min(emb.head_start+emb.part_n_hash_cols,emb.n_hash_cols))
        hashes[0].next_prefetcher=self
        self.verify_enabled=False
        self.current_check=None
        import vllm.models.deepseek_v4_1.common.engram as module
        module._NEXT_PREFETCH_STATS=self
        self.enabled=os.environ.get('DSV41_ENGRAM_NEXT_PREFETCH','0')=='1'
        self.reader=Reader(2)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='engram-next')
        self.future=None
        self.expected=None
        self.reset_stats()

    def reset_stats(self):
        self.stats=dict(submitted=0,skipped_busy=0,rows=0,bytes=0,read_seconds=0.,
                        plan_seconds=0.,stage_seconds=0.,stages=0,errors=0,verified_tokens=0)

    def configure(self, enabled, verify=False):
        if self.future is not None:
            self.future.result()
            self.future=None
        self.expected=None
        self.current_check=None
        self.verify_enabled=verify
        self.enabled=enabled
        self.reset_stats()

    def prepare(self, batch, states, excluded):
        if not self.enabled or not batch.has_prefill:
            return None
        if self.future is not None and not self.future.done():
            self.stats['skipped_busy']+=1
            return None
        choice=next_chunk(batch,states,self.stager.max_tokens,excluded)
        if choice is None or not self.stager.hash_state.ensure_cache():
            return None
        import torch
        from vllm.models.deepseek_v4_1.common.mm_preprocess import image_sentinel_mask
        began=time.perf_counter()
        index,start,n=choice
        state=self.stager.hash_state
        ids=states.all_token_ids.gpu[index,start:start+n]
        depth=state.lookback_depth
        lookback=torch.full((1,depth),-1,dtype=torch.int32,device=ids.device)
        take=min(depth,start)
        if take:
            lookback[0,:take]=states.all_token_ids.gpu[index,start-take:start].flip(0)
        positions=torch.arange(start,start+n,dtype=torch.int64,device=ids.device)
        query=torch.tensor([0,n],dtype=torch.int32,device=ids.device)
        hashes=state(ids,positions,query,image_sentinel_mask(ids),lookback,
                     image_sentinel_mask(lookback),None,None)
        # Separate host buffer: demand staging is never replaced or reused.
        host=torch.empty((n,hashes.shape[1],self.stager.head_end-self.stager.head_start),
                         dtype=torch.int32,device='cpu',pin_memory=True)
        host.copy_(hashes[:,:,self.stager.head_start:self.stager.head_end],non_blocking=True)
        ready=torch.cuda.Event();ready.record();ready.synchronize()
        self.stats['plan_seconds']+=time.perf_counter()-began
        return (states.index_to_req_id[index],start,host)

    def begin_batch(self, batch):
        self.current_check=None
        if self.expected is None or not self.verify_enabled:
            return
        req_id,start,host=self.expected
        self.expected=None
        for i,current_id in enumerate(batch.req_ids):
            if current_id == req_id and int(batch.num_computed_prefill_tokens_np[i]) == start:
                self.current_check=(int(batch.query_start_loc_np[i]),
                    min(host.shape[0],int(batch.num_scheduled_tokens[i])),host)

    def verify_hashes(self, hashes):
        if self.current_check is None:
            return
        import torch
        offset,n,host=self.current_check
        self.current_check=None
        actual=hashes[offset:offset+n,:,self.stager.head_start:self.stager.head_end].cpu()
        if not torch.equal(actual,host[:n]):
            raise RuntimeError('Forecast Engram hashes differ from demand hashes')
        self.stats['verified_tokens']+=n

    def submit(self, plan):
        if plan is None:
            return
        self.expected=plan
        host=plan[2]
        self.stats['submitted']+=1
        self.future=self.pool.submit(self._read,host)

    def _read(self, host):
        import torch
        began=time.perf_counter()
        try:
            jobs=[]
            for engram in self.stager.engrams:
                emb=engram.embed_tokens
                rows=host[:,engram.layer_hash_index].reshape(-1).to(torch.int64)
                owned=(rows>=emb.vocab_start_idx)&(rows<emb.vocab_end_idx)
                rel=torch.unique(rows[owned]-emb.vocab_start_idx).tolist()
                table=emb.disk
                for fd,base,width in [(table.w_fd,table.w_off,table.dim),(table.s_fd,table.s_off,table.sb)]:
                    buf=bytearray(len(rel)*width)
                    jobs.append((fd,base,rel,width,buf))
                    self.stats['bytes']+=len(buf)
                self.stats['rows']+=len(rel)
            # Buffered preads warm the same OS pages consumed by the demand reader.
            # Disposable bytes, no decoded-row cache and no shared staging buffer.
            self.reader.read(jobs,chunk=16)
        except Exception as exc:
            self.stats['last_error']=str(exc)
            self.stats['errors']+=1
            self.enabled=False
        finally:
            self.stats['read_seconds']+=time.perf_counter()-began
