"""Check TP shards against safetensors' independent tensor reader."""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
import torch
from safetensors import safe_open
from expert_store import ExpertStore

model = sys.argv[1]
for rank in (0, 1):
    store = ExpertStore(model, rank=rank, cache_bytes=10*1024**2)
    for layer, expert in ((0, 0), (14, 383), (39, 192)):
        item = store.expert(layer, expert)
        for key, (raw, shape) in item.items():
            name = f'layers.{layer}.ffn.experts.{expert}.{key}'
            path, _, _ = store.index[name]
            with safe_open(str(path), framework='pt', device='cpu') as f:
                tensor = f.get_tensor(name).view(torch.uint8)
                axis = 1 if key.startswith('w2.') else 0
                expected = tensor.chunk(2, dim=axis)[rank].contiguous()
                actual = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape)
                assert torch.equal(actual, expected), (rank, name)
        assert store.resident_bytes <= store.cache_bytes
    misses = store.stats['misses']
    store.expert(39, 192)
    assert store.stats['misses'] == misses
    store.expert(0, 0)
    assert store.stats['misses'] == misses + 1  # evicted expert reloads
    store.close()
print('PASS: both TP ranks, 3 layers, 6 tensors/expert; exact bytes, eviction and cache hit')
