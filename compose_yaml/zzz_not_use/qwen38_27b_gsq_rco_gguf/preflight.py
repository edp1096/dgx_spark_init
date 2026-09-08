#!/usr/bin/env python3
"""Inspect local BF16 inputs without loading tensors or starting GPU services.

This checks safetensors headers, index coverage and payload sizes. It does not
hash the tensor payloads or provide a GSQ-RCO quantization implementation.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import struct


def inspect_model(directory):
    directory = directory.resolve(strict=True)
    config = json.loads((directory / 'config.json').read_text())
    if config.get('quantization_config'):
        raise ValueError(f'{directory}: quantized input config')
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    weights = index['weight_map']
    found, shards = {}, []
    for name in sorted(set(weights.values())):
        if Path(name).name != name:
            raise ValueError(f'Unexpected shard path: {name}')
        file = directory / name
        length = file.stat().st_size
        with file.open('rb') as stream:
            header_length = struct.unpack('<Q', stream.read(8))[0]
            if header_length > min(length - 8, 64 << 20):
                raise ValueError(f'{name}: invalid header length')
            raw = stream.read(header_length)
            header = json.loads(raw)
        intervals = []
        for tensor, value in header.items():
            if tensor == '__metadata__':
                continue
            if tensor in found or weights.get(tensor) != name:
                raise ValueError(f'{tensor}: duplicate or incorrect index entry')
            if value['dtype'] != 'BF16':
                raise ValueError(f'{tensor}: expected BF16, found {value["dtype"]}')
            shape = value['shape']
            if any(not isinstance(d, int) or d < 0 for d in shape):
                raise ValueError(f'{tensor}: invalid shape')
            start, end = value['data_offsets']
            if start < 0 or end - start != math.prod(shape) * 2:
                raise ValueError(f'{tensor}: inconsistent tensor byte count')
            intervals.append((start, end))
            found[tensor] = {'shape': shape, 'dtype': value['dtype']}
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError(f'{name}: gap or overlap in tensor data')
            cursor = end
        if 8 + header_length + cursor != length:
            raise ValueError(f'{name}: truncated or unexpected payload')
        shards.append({'name': name, 'bytes': length,
                       'header_sha256': hashlib.sha256(raw).hexdigest()})
    if set(found) != set(weights):
        raise ValueError('Weight index is incomplete')
    for required in ['tokenizer.json', 'tokenizer_config.json']:
        if not (directory / required).is_file():
            raise ValueError(f'Missing {required}')
    schema = json.dumps(found, sort_keys=True, separators=(',', ':')).encode()
    return {
        'path': str(directory), 'model_type': config.get('model_type'),
        'tensor_count': len(found), 'dtype_counts': dict(collections.Counter(
            value['dtype'] for value in found.values())),
        'weight_bytes': sum(s['bytes'] for s in shards),
        'tensor_schema_sha256': hashlib.sha256(schema).hexdigest(),
        'has_mtp': any('mtp' in name for name in found),
        'has_vision': any('visual' in name for name in found),
        'shards': shards, 'validation': 'index/header/payload-length checks passed',
        'payload_checksum_verified': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official', required=True, type=Path)
    parser.add_argument('--abliterated', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    official = inspect_model(args.official)
    abliterated = inspect_model(args.abliterated)
    if official['tensor_schema_sha256'] != abliterated['tensor_schema_sha256']:
        raise ValueError('Official and abliterated tensor schemas differ')
    report = {'official': official, 'abliterated': abliterated,
              'matching_tensor_schema': True,
              'scope': 'BF16 input inspection only; GGUF generation tooling is not included'}
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
