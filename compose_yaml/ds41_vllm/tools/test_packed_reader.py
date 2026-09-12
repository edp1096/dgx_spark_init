"""Stress multiple staging windows and verify slot bytes after full eviction."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys,torch
from expert_store import ExpertStore
from b12x_slots import SlotLayer,stats
store=ExpertStore(sys.argv[1],rank=0)
cache=SlotLayer(store,0,6)
for begin in (0,16):
    wanted=list(range(begin,begin+16));cache.ensure(wanted)
    for expert in wanted:
        data=cache.read(expert);offset=0;slot=cache.used[expert]
        for part,desc in zip(cache.parts,cache.meta['layout']):
            length=desc['bytes']
            actual=part[slot].cpu().contiguous().view(torch.uint8).reshape(-1)
            assert torch.equal(actual,data[offset:offset+length]),expert
            offset+=length
    before=stats()['packed_read_bytes'];cache.ensure(wanted)
    assert stats()['packed_read_bytes']==before
print('BOUNDED_READER_AND_FULL_EVICTION_PASS',stats(),flush=True)
