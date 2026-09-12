"""Frequency-based startup loading into existing expert slots; no routing changes."""
import hashlib
import json
import os
import time
from pathlib import Path


def read_profile(path, revision, layout):
    raw = Path(path).read_bytes()
    profile = json.loads(raw)
    if (profile.get('schema'), profile.get('revision'), profile.get('format')) != (1, revision, layout):
        raise ValueError('Expert preload profile checkpoint/layout mismatch')
    layers = profile['layers']
    if set(layers) != {str(i) for i in range(40)}:
        raise ValueError('Expert preload requires all 40 target layers')
    for ids in layers.values():
        if not isinstance(ids, list) or not ids or any(type(e) is not int or not 0 <= e < 384 for e in ids) or len(set(ids)) != len(ids):
            raise ValueError('Invalid/duplicate expert ID in preload profile')
    return layers, hashlib.sha256(raw).hexdigest()


def seed(count, verify=False):
    import torch
    import b12x_slots as slots
    count = int(count)
    if not 0 <= count <= 224:
        raise ValueError('Preload count must be 0..224')
    if set(range(40)) - slots._layers.keys():
        raise RuntimeError('Preload must run after all target layers are initialized')
    first = slots._layers[0]
    profile, digest = read_profile(Path(__file__).with_name('expert-hot-profile.json'), first.meta['revision'], first.meta['format'])
    for layer in range(40):
        cache = slots._layers[layer]
        if cache.count < count or cache.meta['revision'] != first.meta['revision'] or cache.meta['format'] != first.meta['format']:
            raise ValueError('Inconsistent target cache/profile configuration')
    torch.cuda.synchronize()
    start = time.monotonic()
    before = slots.stats()
    loaded = checked = 0
    for layer in range(40):
        cache = slots._layers[layer]
        cache.reset()
        ids = list(reversed(profile[str(layer)][:count]))
        if ids:
            cache.ensure(ids)
            loaded += len(ids)
            if verify:
                for expert in dict.fromkeys((ids[0], ids[-1])):
                    cache.verify_slot(expert, cache.used[expert])
                    checked += 1
        cache.last_event.record(torch.cuda.current_stream())
        cache.last_stream = torch.cuda.current_stream().cuda_stream
    torch.cuda.synchronize()
    after = slots.stats()
    result = {'rank': first.rank, 'count': count, 'loaded': loaded, 'verified_slots': checked,
              'seconds': time.monotonic() - start, 'profile_sha256': digest,
              'read_bytes': after['packed_read_bytes'] - before['packed_read_bytes'],
              'gpu_cache_bytes': after['gpu_cache_bytes']}
    print('EXPERT_PRELOAD ' + json.dumps(result), flush=True)
    return result


def startup():
    count = int(os.environ.get('DSV41_PRELOAD_COUNT', '0'))
    if count:
        return seed(count)


def benchmark(action='stats', count='0', verify='0'):
    if os.environ.get('DSV41_BENCH_CONTROL') != '1':
        raise RuntimeError('Expert preload benchmark RPC disabled')
    import torch
    import b12x_slots as slots
    torch.cuda.synchronize()
    if action == 'seed':
        result = seed(int(count), verify == '1')
        for key in slots._stats:
            slots._stats[key] = 0
        return result
    if action == 'stats':
        return slots.stats()
    raise ValueError('Unknown expert preload benchmark action')
