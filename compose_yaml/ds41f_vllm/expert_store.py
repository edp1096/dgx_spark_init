"""Read original MXFP4 expert bytes without mapping/loading whole weight shards."""
from __future__ import annotations
import argparse
from collections import OrderedDict, defaultdict
import json
import os
from pathlib import Path
import re
import struct
import time

EXPERT = re.compile(r'^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$')

class ExpertStore:
    def __init__(self, model, rank=0, tp=2, cache_bytes=0):
        if tp != 2 or rank not in (0, 1):
            raise ValueError('This prototype supports TP2 ranks 0 and 1 only')
        self.model = Path(model)
        self.rank, self.tp = rank, tp
        self.index = {}
        self.fds = {}
        self.cache = OrderedDict()
        self.cache_bytes = cache_bytes
        self.resident_bytes = 0
        self.stats = dict(hits=0, misses=0, read_bytes=0, read_seconds=0.0)
        self.totals = defaultdict(int)
        for path in sorted(self.model.glob('*.safetensors')):
            with path.open('rb') as f:
                raw = f.read(8)
                size = struct.unpack('<Q', raw)[0]
                if size > 64 * 1024 * 1024:
                    raise ValueError(f'Invalid header: {path}')
                header = json.loads(f.read(size))
                if os.environ.get("DSV41_RELEASE_CHECKPOINT_CACHE") == "1":
                    os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            for name, meta in header.items():
                if name == '__metadata__':
                    continue
                start, end = meta['data_offsets']
                if end < start or 8 + size + end > path.stat().st_size:
                    raise ValueError(f'Invalid offsets: {name}')
                group = 'experts' if EXPERT.match(name) else ('engram_tables' if '.engram.embed.' in name else 'other')
                self.totals[group] += end-start
                if EXPERT.match(name):
                    self.index[name] = (path, 8+size+start, meta)

    def _read(self, path, offset, length):
        fd = self.fds.get(path)
        if fd is None:
            fd = self.fds[path] = os.open(path, os.O_RDONLY)
        chunks = []
        start = time.monotonic()
        remaining = length
        while remaining:
            block = os.pread(fd, remaining, offset)
            if not block:
                raise EOFError(f'Short tensor read: {path} at {offset}')
            chunks.append(block)
            offset += len(block)
            remaining -= len(block)
        self.stats['read_bytes'] += length
        self.stats['read_seconds'] += time.monotonic()-start
        return b''.join(chunks)

    def tensor(self, name):
        path, offset, meta = self.index[name]
        rows, cols = meta['shape']
        if meta['dtype'] not in ('I8', 'U8', 'F8_E8M0'):
            raise ValueError(f'Unsupported expert dtype {meta}')
        if '.w2.' in name:
            # w2 is split along its packed columns. Read once rather than
            # issuing 5120 tiny preads; only this rank's bytes survive.
            if cols % self.tp:
                raise ValueError(f'Unaligned TP columns: {name}')
            width = cols // self.tp
            raw = self._read(path, offset, rows*cols)
            begin = self.rank * width
            part = b''.join(raw[r*cols+begin:r*cols+begin+width] for r in range(rows))
            return part, (rows, width)
        if rows % self.tp:
            raise ValueError(f'Unaligned TP rows: {name}')
        height = rows // self.tp
        return self._read(path, offset+self.rank*height*cols, height*cols), (height, cols)

    def expert(self, layer, expert, family='layers'):
        key = (family, layer, expert)
        if key in self.cache:
            self.stats['hits'] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.stats['misses'] += 1
        tensors = {f'{w}.{kind}': self.tensor(f'{family}.{layer}.ffn.experts.{expert}.{w}.{kind}')
                   for w in ('w1', 'w2', 'w3') for kind in ('weight', 'scale')}
        size = sum(len(raw) for raw, shape in tensors.values())
        if size <= self.cache_bytes:
            while self.resident_bytes + size > self.cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.resident_bytes -= sum(len(raw) for raw, shape in old.values())
            self.cache[key] = tensors
            self.resident_bytes += size
        return tensors

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    s = ExpertStore(args.model, rank=args.rank, cache_bytes=32*1024**2)
    print(json.dumps({'bytes': dict(s.totals), 'expert_tensor_count': len(s.index)}, indent=2))
    if args.probe:
        for e in [0, 1, 0]:
            s.expert(0, e)
        print(json.dumps(s.stats | {'resident_bytes': s.resident_bytes}, indent=2))
    s.close()
