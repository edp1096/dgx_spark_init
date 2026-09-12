"""Overlap actual-route reads into a separate bank with resident expert compute.

One 160-expert bank per CUDA stream is shared across layers. All reads target
storage distinct from the resident bank; no reader can overwrite a live kernel's
weights or scales. The bank is copied into LRU slots only after both executions.
"""
from collections import Counter
from contextlib import nullcontext
import os,struct
import torch
from b12x_layout import blank,tensors
import b12x_slots
import expert_io

_banks={}

def enabled():
    if os.environ.get('DSV41_BENCH_CONTROL')=='1':
        import cache_control
        return cache_control.routed_pipeline
    return os.environ.get('DSV41_ROUTED_PIPELINE','0')=='1'

def bank_for(cache):
    key=(torch.cuda.current_device(),torch.cuda.current_stream().cuda_stream,cache.topk)
    if key not in _banks:
        from b12x.loader._pool import shared_pool
        bank=object.__new__(b12x_slots.SlotLayer)
        bank.count=bank.capacity=cache.meta['experts']-cache.count
        bank.topk=cache.topk;bank.calls={};bank.prefetch_pending={}
        with shared_pool(allocation='registered'): bank.prepared=blank(bank.count)
        bank.parts=tensors(bank.prepared)
        _banks[key]=bank
    bank=_banks[key]
    if bank.count!=cache.meta['experts']-cache.count:
        raise ValueError('Routed staging requires a uniform target cache')
    return bank

def prime(cache,x,weights,shape):
    # GLOBAL graph capture cannot run concurrently with native reader fences.
    limit,_=b12x_slots.execution_options()
    for begin in range(0,x.shape[0],limit):
        end=min(begin+limit,x.shape[0]);n=end-begin
        call=cache.execution(n)
        if os.environ.get('DSV41_EXPERT_GRAPHS','1')=='1' and call['graph'] is None:
            cache.execute(x[begin:end],torch.zeros_like(weights[begin:end]),[-1]*(n*shape[1]),(n,shape[1]))

def run(cache,x,weights,ids,needed,on_load_start):
    if not cache.direct or expert_io.mode()!='batch_overlap' or cache.prefetch_pending:
        raise ValueError('Routed pipeline requires native batch overlap without prediction')
    unique=set(needed)
    frequency=Counter(needed[-64*cache.topk:])
    resident=sorted(unique.intersection(cache.used),key=lambda e:(frequency[e],e))
    missing=sorted(unique-set(resident),key=lambda e:(frequency[e],e))
    fill=cache.count-len(resident)
    main_group=resident+missing[:fill]
    staged_group=missing[fill:]
    bank=bank_for(cache)
    assert 0<len(staged_group)<=bank.count
    slots=cache.ensure(main_group,on_load_start)
    main_map=dict(zip(main_group,slots))
    stage_map={e:i for i,e in enumerate(staged_group)}
    prime(cache,x,weights,ids.shape);prime(bank,x,weights,ids.shape)
    records=bytearray()
    for expert,slot in stage_map.items():
        offset=b12x_slots.HEADER_BYTES+expert*cache.meta['record_bytes']
        for part,desc in zip(bank.parts,cache.meta['layout']):
            records.extend(struct.pack('<8Q',cache.fd,offset,desc['bytes'],part[slot].data_ptr(),0,1,0,0))
            offset+=desc['bytes']
    future=expert_io.submit_batch(records,expert_io.read_stream())
    try:
        if on_load_start is not None: on_load_start()
        # Consume the borrowed graph result on this stream before bank replay.
        output=cache.execute(x,weights,[main_map.get(e,-1) for e in needed],ids.shape).clone()
        future.result()
        output.add_(bank.execute(x,weights,[stage_map.get(e,-1) for e in needed],ids.shape))
        # Keep the new experts for subsequent chunks/decode, without another read.
        retained=[]
        for expert in staged_group:
            if cache.free: target=cache.free.pop()
            else:
                victim=next(e for e in cache.used if e not in stage_map)
                target=cache.used.pop(victim)
            retained.append(target)
        # Four indexed GPU copies avoid 640 per-expert copy launches.
        target_ids=torch.tensor(retained,device=x.device,dtype=torch.int64)
        for dest,src in zip(cache.parts,bank.parts):
            dest.index_copy_(0,target_ids,src[:len(staged_group)])
        for expert,target in zip(staged_group,retained): cache.used[expert]=target
    finally:
        # Reset/cancellation may reuse either bank only after every writer exits.
        future.result()
    size=len(staged_group)*cache.meta['record_bytes']
    b12x_slots._stats['slot_misses']+=len(staged_group)
    b12x_slots._stats['packed_read_bytes']+=size
    b12x_slots._stats['demand_read_bytes']+=size
    b12x_slots._stats['io_batches']+=1
    b12x_slots._stats['routed_pipeline_batches']+=1
    b12x_slots._stats['routed_pipeline_bytes']+=size
    return output

def stats():
    return {'staging_bytes':sum(t.numel()*t.element_size() for b in _banks.values() for t in b.parts),
            'staging_banks':len(_banks)}
