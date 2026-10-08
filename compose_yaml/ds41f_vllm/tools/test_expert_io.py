"""Exact slot bytes/output, eviction, shared overlap, stream ordering and failures."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import os,sys,threading
import torch
import b12x_slots,expert_io
from expert_store import ExpertStore

torch.manual_seed(934)
store=ExpertStore(sys.argv[1],rank=int(os.environ.get('DSV41_TEST_RANK','0')))
shared=torch.nn.Sequential(torch.nn.Linear(5120,128,bias=False),torch.nn.SiLU(),torch.nn.Linear(128,5120,bias=False)).cuda().bfloat16()

def set_mode(mode):
    expert_io.DEFAULT_MODE=mode
    expert_io.BENCH_CONTROL=False
    for layer in b12x_slots._layers.values(): layer.reset()

def run(mode,x,ids,weights):
    set_mode(mode)
    result=None;calls=0
    def overlap():
        nonlocal result,calls
        if result is not None:return False
        result=shared(x).float();calls+=1
        return True
    routed=b12x_slots.apply(store,0,x,weights,ids,10,on_load_start=overlap if expert_io.overlaps() else None)
    out=routed+(result if result is not None else shared(x).float())
    torch.cuda.synchronize()
    assert calls==(1 if expert_io.overlaps() else 0)
    cache=b12x_slots._layers[0]
    for expert,slot in cache.used.items():cache.verify_slot(expert,slot)
    return out.clone(),list(cache.used)

with torch.inference_mode():
    for n in (1,6,17):
        x=torch.randn(n,5120,device='cuda',dtype=torch.bfloat16)*.1
        ids=(torch.arange(n*2,device='cuda',dtype=torch.int32).reshape(n,2)+355)%384
        weights=torch.rand(n,2,device='cuda')
        expected,order=run('serial',x,ids,weights)
        for mode in ('overlap','batch','batch_overlap'):
            actual,actual_order=run(mode,x,ids,weights)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            assert actual_order==order
            print('EXPERT_IO_EXACT_PASS',n,mode,flush=True)
    cache=b12x_slots._layers[0]
    # Work queued AFTER the reader fence must not block O_DIRECT. A gated
    # reader starts only once the GPU delay is queued; this catches accidentally
    # synchronizing the main stream instead of the reader stream.
    for mode in ('overlap','batch_overlap'):
        set_mode(mode)
        expert_io.batch_executor()
        release=threading.Event();original_read=cache.read_into;original_submit=expert_io.submit_batch
        def delayed_read(*args):
            assert release.wait(5)
            return original_read(*args)
        def delayed_submit(records,stream):
            native,executor=expert_io.batch_executor()
            def read():
                assert release.wait(5)
                return native.batch_execute(executor,records,stream)
            return expert_io._submitter.submit(read)
        cache.read_into=delayed_read;expert_io.submit_batch=delayed_submit
        end=torch.cuda.Event()
        def during_read():
            torch.cuda._sleep(1_000_000_000)
            end.record()
            release.set()
            return True
        try:
            cache.ensure([0,1,382,383],during_read)
            assert not end.query(),'I/O waited for later independent GPU work'
            end.synchronize()
            for expert,slot in cache.used.items():cache.verify_slot(expert,slot)
        finally:
            cache.read_into=original_read;expert_io.submit_batch=original_submit
        print('EXPERT_IO_CONCURRENT_PASS',mode,flush=True)
    # Subsequent use on another stream retains exact slot contents and outputs.
    set_mode('batch_overlap')
    x=torch.randn(6,5120,device='cuda',dtype=torch.bfloat16)*.1
    ids=torch.arange(12,device='cuda',dtype=torch.int32).reshape(6,2)
    weights=torch.rand(6,2,device='cuda')
    expected,_=run('batch_overlap',x,ids,weights)
    other=torch.cuda.Stream();other.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(other):actual,_=run('batch_overlap',x,ids,weights)
    torch.cuda.current_stream().wait_stream(other)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    print('EXPERT_IO_CROSS_STREAM_PASS',flush=True)
    for mode in ('overlap','batch_overlap'):
        set_mode(mode)
        def fail():raise ValueError('injected callback failure')
        try:cache.ensure([2,3],fail)
        except ValueError as error:assert str(error)=='injected callback failure'
        else:raise AssertionError('Callback error was ignored')
        actual,_=run(mode,x,ids,weights)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        print('EXPERT_IO_CALLBACK_FAILURE_DRAIN_PASS',mode,flush=True)
    set_mode('batch')
    original=cache.fd;cache.fd=999999
    try:
        try:cache.ensure([4,5])
        except RuntimeError:pass
        else:raise AssertionError('I/O error was ignored')
    finally:cache.fd=original
    print('EXPERT_IO_READ_ERROR_PASS',flush=True)
print(b12x_slots.stats(),flush=True)
