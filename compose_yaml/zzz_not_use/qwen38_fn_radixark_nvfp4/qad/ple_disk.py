"""TP1 immutable checkpoint PLE rows, with bounded eager io_uring staging.

SGLang hashes PLE IDs inside its decode graph. Those graphs use the original
safetensors CPU mappings through GB10 UVA; eager prefills use B12X row I/O.
No checkpoint tensor is copied or rewritten. This adapter does not import vLLM.
"""
import json
import math
import bisect
import os
from pathlib import Path
import re
import struct
import threading

import torch
import triton
import triton.language as tl

from nvfp4_ple import gather


class CheckpointRssTrimmer:
    """Bound graph-decode UVA residency across all immutable PLE mappings.

io_uring prefill staging is bounded already. CUDA graph replay can still
fault original file pages in during decode, so a Python gather hook is not
sufficient to limit their accumulation. Like SGLang's file-table trimmer,
this daemon drops only clean file mapping pages, in small pieces.
"""
    def __init__(self, tensors, paths, budget, interval=10):
        # Retain the mappings until the worker is closed. Never apply DONTNEED
        # to anonymous CPU allocations: it would erase their contents.
        self.tensors=tuple(tensors)
        paths={str(Path(p).resolve()) for p in paths}
        mappings=[]
        with open('/proc/self/maps') as source:
            for line in source:
                fields=line.strip().split(maxsplit=5)
                if len(fields)!=6 or fields[5] not in paths: continue
                lo,hi=(int(x,16) for x in fields[0].split('-'))
                mappings.append((lo,hi))
        needed=set()
        for tensor in self.tensors:
            lo,hi=tensor.data_ptr(),tensor.data_ptr()+tensor.nbytes
            covering=[(a,b) for a,b in mappings if a<=lo and hi<=b]
            if len(covering)!=1:
                raise ValueError('PLE trimmer requires an immutable checkpoint file mapping')
            needed.add(covering[0])
        # A safe_open mapping also covers adjacent checkpoint tensors. All
        # source data is immutable, and H2D loading has finished by this point.
        # Trim/count whole file VMAs so unrelated resident source pages do not
        # permanently inflate an otherwise unmeasurable subrange RSS budget.
        self.ranges=sorted(needed)
        self.starts=[lo for lo,hi in self.ranges]
        self.budget=budget
        self.interval=interval
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self._loop,name='qad-ple-rss-trim',daemon=True)

    def resident_bytes(self):
        total=0; relevant=False
        # Read smaps once for all 256 planes; count each overlapping VMA once.
        with open('/proc/self/smaps') as source:
            for line in source:
                match=re.match(r'^([0-9a-f]+)-([0-9a-f]+) ',line)
                if match:
                    lo,hi=(int(x,16) for x in match.groups())
                    i=bisect.bisect_left(self.starts,hi)-1
                    relevant=i>=0 and self.ranges[i][1]>lo
                elif relevant and line.startswith('Rss:'):
                    total+=int(line.split()[1])*1024
        return total

    def trim_once(self, force=False):
        from sglang.srt.models.qwen4_exp_ple_table import _madvise, _MADV_DONTNEED
        before=self.resident_bytes()
        if not force and before<=self.budget: return 0
        for lo,hi in self.ranges:
            for address in range(lo,hi,64*1024**2):
                if self.stop.is_set(): return 0
                if not _madvise(address,min(64*1024**2,hi-address),_MADV_DONTNEED):
                    raise RuntimeError('Unable to trim checkpoint PLE mapping')
                self.stop.wait(.005)
        return max(0,before-self.resident_bytes())

    def _loop(self):
        import logging
        while not self.stop.wait(self.interval):
            try:
                freed=self.trim_once()
                if freed: logging.getLogger(__name__).info('QAD PLE mapping trim freed %.2f GiB',freed/2**30)
            except Exception:
                logging.getLogger(__name__).exception('QAD PLE mapping trim failed')

    def close(self):
        self.stop.set()
        if self.thread.is_alive(): self.thread.join(timeout=2)


@triton.jit
def _lookup(weights, scales, factor, ids, out, ROWS: tl.constexpr,
            SHARD_ROWS: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    index = tl.load(ids + row)
    valid = (index >= 0) & (index < ROWS)
    index = tl.where(valid, index, 0)
    shard = index // SHARD_ROWS
    local = index % SHARD_ROWS
    wp = tl.load(weights + shard).to(tl.pointer_type(tl.uint8))
    sp = tl.load(scales + shard).to(tl.pointer_type(tl.float8e4nv))
    col = tl.arange(0, B)
    mask = valid & (col < D)
    packed = tl.load(wp + local * (D // 2) + col // 2, mask, 0)
    bits = (packed >> ((col % 2) * 4)) & 15
    mag = bits & 7
    value = tl.where(mag < 2, mag * .5,
            tl.where(mag < 4, 1 + (mag - 2) * .5,
            tl.where(mag < 6, 2 + (mag - 4), 4 + (mag - 6) * 2)))
    value = tl.where((bits & 8) != 0, -value, value)
    scale = tl.load(sp + local * (D // 16) + col // 16, mask, 0.0).to(tl.float32)
    value = value * (scale * tl.load(factor))
    tl.store(out + row * D + col, tl.where(valid, value, 0), col < D)


def checkpoint_sources(directory):
    """Read only indexed PLE headers, preserving their exact file byte offsets."""
    directory = Path(directory).resolve()
    index = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = '.ple.ple_embedding.ngram_embedding.'
    wanted = {name: file for name, file in index.items()
              if prefix in name and '.shard_' in name}
    if not wanted:
        raise ValueError('Checkpoint has no indexed packed PLE shards')
    headers = {}
    result = {}
    for name, file in wanted.items():
        path = directory / file
        if file not in headers:
            with path.open('rb') as source:
                length = struct.unpack('<Q', source.read(8))[0]
                if not 0 < length <= 64 * 1024**2:
                    raise ValueError('Invalid safetensors header length')
                headers[file] = (length + 8, json.loads(source.read(length)))
        base, header = headers[file]
        item = header[name]
        lo, hi = item['data_offsets']
        if lo < 0 or hi <= lo or base + hi > path.stat().st_size:
            raise ValueError(f'Invalid PLE file extent: {name}')
        suffix = name.split(prefix, 1)[1]
        if suffix in result:
            raise ValueError('Multiple PLE tables are not supported by this TP1 adapter')
        result[suffix] = (str(path), base + lo, item['shape'], item['dtype'])
    return result


class CheckpointRows:
    def __init__(self, rows, dim, shards, directory):
        self.rows, self.dim, self.shards = rows, dim, shards
        self.shard_rows = math.ceil(rows / shards)
        self.sources = checkpoint_sources(directory)
        self.tensors = {}
        self.cache = None
        self.capacity = 0
        self.trimmer = None

    def load(self, suffix, value):
        path, offset, shape, dtype = self.sources[suffix]
        expected_dtype = 'U8' if suffix.endswith('.weight') else 'F8_E4M3'
        if value.device.type != 'cpu' or not value.is_contiguous():
            raise ValueError('PLE source must remain a contiguous CPU checkpoint mapping')
        if list(value.shape) != shape or dtype != expected_dtype:
            raise ValueError(f'PLE source metadata mismatch: {suffix}')
        # Keep the original safetensors storage alive after the loader closes it.
        self.tensors[suffix] = value

    def finish(self):
        expected = {f'shard_{i}.{kind}' for i in range(self.shards)
                    for kind in ('weight', 'weight_scale')}
        if set(self.tensors) != expected or set(self.sources) != expected:
            raise ValueError('Incomplete checkpoint-backed PLE table')
        self.weight_ptrs = torch.tensor([self.tensors[f'shard_{i}.weight'].data_ptr()
                                        for i in range(self.shards)], device='cuda', dtype=torch.int64)
        self.scale_ptrs = torch.tensor([self.tensors[f'shard_{i}.weight_scale'].data_ptr()
                                       for i in range(self.shards)], device='cuda', dtype=torch.int64)
        budget=float(os.environ.get('SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB','4'))
        if not math.isfinite(budget) or budget<0: raise ValueError('Invalid PLE RSS budget')
        if budget>0:
            self.trimmer=CheckpointRssTrimmer(self.tensors.values(),
                [source[0] for source in self.sources.values()],int(budget*2**30))
            self.trimmer.thread.start()

    def _cache_for(self, count, device):
        if count <= self.capacity:
            return self.cache
        from b12x.sequence._shared.disk_table import DiskRowCache
        capacity = triton.next_power_of_2(count)
        cache = DiskRowCache(device=device, max_lookups=capacity,
            table_rows=self.rows, shard_start=0, shard_end=self.rows,
            shard_rows=self.shard_rows, weight_row_bytes=self.dim // 2,
            scale_row_bytes=self.dim // 16)
        for i in range(self.shards):
            for kind in ('weight', 'weight_scale'):
                path, offset, _, _ = self.sources[f'shard_{i}.{kind}']
                cache.add_shard(i, path, offset, scale=kind == 'weight_scale')
        cache.freeze()
        self.cache, self.capacity = cache, capacity
        return cache

    def lookup(self, ids, factor, out=None):
        shape = (*ids.shape, self.dim)
        if out is None:
            out = torch.empty(shape, dtype=torch.bfloat16, device=ids.device)
        if (out.shape != shape or out.dtype != torch.bfloat16
                or out.device != ids.device or not out.is_contiguous()):
            raise ValueError('Invalid PLE output buffer')
        if ids.numel() == 0:
            return out
        # Small decode/verify batches use the same path during warmup/capture.
        # io_uring cannot be part of a CUDA graph: its CPU work would not replay.
        if ids.numel() <= 256 or torch.cuda.is_current_stream_capturing():
            _lookup[(ids.numel(),)](self.weight_ptrs, self.scale_ptrs, factor,
                ids, out, self.rows, self.shard_rows, self.dim,
                triton.next_power_of_2(self.dim))
        else:
            cache = self._cache_for(ids.numel(), ids.device)
            with cache.transaction():
                cache.read_rows(ids, ids.numel())
                # The native reader writes rows in request order, including
                # zero rows for invalid IDs; compact IDs select those rows.
                compact = torch.arange(ids.numel(), device=ids.device).reshape(ids.shape)
                gather(cache.weight, cache.scale.view(torch.float8_e4m3fn),
                       factor, compact, out=out)
        return out
