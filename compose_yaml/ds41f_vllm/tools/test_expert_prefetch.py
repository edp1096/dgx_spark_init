"""Exercise real direct reads, late arrivals, unused forecasts and GPU reuse."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import os,sys,time,threading
import torch
from expert_store import ExpertStore
from b12x_slots import SlotLayer,stats,_layers

torch.manual_seed(53)
store=ExpertStore(sys.argv[1],rank=0)
os.environ['DSV41_PREFETCH_TEST']='0'
reference=SlotLayer(store,0,2)
os.environ['DSV41_PREFETCH_TEST']='1'
cache=SlotLayer(store,0,2);_layers[0]=cache
assert cache.capacity==cache.count+2
x=torch.randn(2,5120,device='cuda').bfloat16()*.1
weights=torch.tensor([[.3,.7],[.6,.4]],device='cuda')
def run(ids):
    routes=torch.tensor(ids,device='cuda',dtype=torch.int32)
    expected=reference.run(x,weights,routes).clone()
    actual=cache.run(x,weights,routes).clone()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert len(cache.used)==4
    assert len(set(cache.used.values()))==4
    assert not set(cache.used.values())&set(cache.prefetch_free)

run([[0,1],[2,3]])
before=dict(cache.used)
cache.start_prefetch([4,5])
for item in cache.prefetch_pending.values(): item['future'].result()
assert dict(cache.used)==before, 'A forecast evicted a resident expert'
run([[4,1],[5,0]])
assert stats()['prefetch_consumed']==2 and stats()['prefetch_ready']==2
print('READY_PREFETCH_PROMOTION_AND_EXACT_OUTPUT_PASS',flush=True)

read=cache.read_into
other=torch.cuda.Stream();other.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(other): run([[4,1],[5,0]])
torch.cuda.current_stream().wait_stream(other)
def delayed(expert,slot,stream):
    time.sleep(.08)
    return read(expert,slot,stream)
cache.read_into=delayed
cache.start_prefetch([6,7])
other.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(other): run([[6,1],[7,0]])
torch.cuda.current_stream().wait_stream(other)
assert stats()['prefetch_late']>=1
print('LATE_PREFETCH_CROSS_STREAM_PASS',flush=True)

before=dict(cache.used)
started={e:threading.Event() for e in (8,9)}
release=threading.Event()
def held(expert,slot,stream):
    started[expert].set()
    assert release.wait(5)
    return read(expert,slot,stream)
cache.read_into=held
routes=torch.tensor([[6,1],[7,0]],device='cuda',dtype=torch.int32)
expected=cache.run(x,weights,routes).clone()
call=cache.execution(2)
cache.start_prefetch([8,9])
assert all(event.wait(5) for event in started.values())
release.set()
for _ in range(200): call['graph'].replay()
torch.cuda.synchronize()
torch.testing.assert_close(call['out'][:2],expected,rtol=0,atol=0)
cache.retire_prefetch(wait=True)
assert dict(cache.used)==before
assert len(cache.prefetch_free)==2
assert stats()['prefetch_unused']>=2
print('UNUSED_FORECAST_NEVER_EVICTS_OR_CORRUPTS_RESIDENTS_PASS',flush=True)

cache.read_into=delayed
cache.start_prefetch([10,11])
cache.reset()
assert not cache.prefetch_pending and not cache.used
assert len(cache.free)==4 and len(cache.prefetch_free)==2
run([[0,1],[2,3]])
print('RESET_DRAINS_READS_BEFORE_SLOT_REUSE_PASS',flush=True)

def broken(expert,slot,stream): raise OSError('injected direct read failure')
cache.read_into=broken
cache.start_prefetch([12])
try: cache.ensure([12,0])
except OSError as error: assert 'injected' in str(error)
else: raise AssertionError('Failed prefetch was published as valid weights')
print('READ_FAILURE_PROPAGATION_PASS',flush=True)
print(stats(),flush=True)
