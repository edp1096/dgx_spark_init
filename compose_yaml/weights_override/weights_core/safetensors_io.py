"""Bounded-memory safetensors I/O. Never downloads or modifies source files."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

SIZES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
         'I16': 2, 'U16': 2, 'BF16': 2, 'F16': 2, 'I32': 4, 'U32': 4,
         'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}
CHUNK = 4 * 1024 * 1024


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_header(path):
    path = Path(path)
    with path.open('rb') as stream:
        prefix = stream.read(8)
        size = int.from_bytes(prefix, 'little')
        if len(prefix) != 8 or not 0 < size < 100_000_000:
            raise ValueError(f'Invalid safetensors header length: {path}')
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError(f'Truncated header: {path}')
        header = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(header, dict):
        raise ValueError('Header must be an object')
    payload = path.stat().st_size - 8 - size
    intervals = []
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        dtype, shape, offsets = tensor['dtype'], tensor['shape'], tensor['data_offsets']
        if dtype not in SIZES:
            raise ValueError(f'Unsupported dtype: {dtype}')
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f'Invalid shape: {name}')
        if len(offsets) != 2 or any(type(d) is not int for d in offsets):
            raise ValueError(f'Invalid offsets: {name}')
        begin, end = offsets
        if not 0 <= begin <= end <= payload or end - begin != math.prod(shape) * SIZES[dtype]:
            raise ValueError(f'Invalid tensor length: {name}')
        intervals.append((begin, end))
    cursor = 0
    for begin, end in sorted(intervals):
        if begin != cursor:
            raise ValueError('Overlapping or non-contiguous tensors')
        cursor = end
    if cursor != payload:
        raise ValueError('Unindexed payload bytes')
    return header, 8 + size


def chunks(path, offset, length):
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        while length:
            data = stream.read(min(CHUNK, length))
            if not data:
                raise ValueError(f'Truncated tensor: {path}')
            length -= len(data)
            yield data


class Checkpoint:
    def __init__(self, directory):
        self.root = Path(directory).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError('Expected a local checkpoint directory')
        self.tensors = {}
        index_file = self.root / 'model.safetensors.index.json'
        expected = None
        if index_file.is_file():
            expected = json.loads(index_file.read_text(), object_pairs_hook=_unique)['weight_map']
            shards = sorted(set(expected.values()))
        else:
            shards = sorted(p.name for p in self.root.glob('*.safetensors'))
        if not shards:
            raise ValueError(f'No local safetensors: {self.root}')
        for shard in shards:
            # HF snapshot symlinks are legitimate; index path traversal is not.
            if Path(shard).name != shard or not shard.endswith('.safetensors'):
                raise ValueError(f'Invalid shard filename: {shard}')
            header, start = read_header(self.root / shard)
            for name, tensor in header.items():
                if name == '__metadata__':
                    continue
                if name in self.tensors:
                    raise ValueError(f'Duplicate tensor: {name}')
                self.tensors[name] = {**tensor, 'shard': shard, 'start': start}
        if expected is not None and expected != {k: v['shard'] for k, v in self.tensors.items()}:
            raise ValueError('Index/header tensor mismatch')

    def blocks(self, name):
        tensor = self.tensors[name]
        begin, end = tensor['data_offsets']
        return chunks(self.root / tensor['shard'], tensor['start'] + begin, end - begin)

    def digest(self, name):
        digest = hashlib.sha256()
        for block in self.blocks(name):
            digest.update(block)
        return digest.hexdigest()


def replace_shard(source, destination, replacements):
    """Copy a shard and replace exact-layout payloads from (path, offset, length).

    Quantizers must supply weight AND scale replacements together. This primitive
    does not choose quantization recipes or certify calibration correctness.
    """
    source, destination = Path(source), Path(destination)
    header, start = read_header(source)
    for name, (path, offset, length) in replacements.items():
        if name == '__metadata__' or name not in header:
            raise ValueError(f'Unknown tensor: {name}')
        begin, end = header[name]['data_offsets']
        if length != end - begin or offset < 0 or Path(path).stat().st_size < offset + length:
            raise ValueError(f'Replacement length mismatch: {name}')
    # Exclusive create also rejects symlinks/hardlinks to source and existing output.
    created = False
    try:
        with destination.open('xb') as output:
            created = True
            with source.open('rb') as original:
                shutil.copyfileobj(original, output, CHUNK)
            for name, (path, offset, length) in replacements.items():
                output.seek(start + header[name]['data_offsets'][0])
                for block in chunks(path, offset, length):
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if read_header(destination) != (header, start):
            raise ValueError('Output header changed')
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
