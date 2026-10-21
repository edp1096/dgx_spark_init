"""Derive a TensorFold vision cache from only the immutable EXL3 vision tensors."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from tensorfold.vision.qwen_checkpoint import vision_tensors, DTYPES
from tensorfold.vision.exl3_convert import convert_tensors, VERSION


def prepare(model, cache):
    model, cache = Path(model), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / 'exl3-vision-f16.safetensors'
    config_bytes = (model / 'config.json').read_bytes()
    raw = json.loads(config_bytes)
    sources = vision_tensors(model)
    digest = hashlib.sha256(config_bytes)
    tensors = {}
    for name, (path, info, begin) in sorted(sources.items()):
        start, end = info['data_offsets']
        with path.open('rb') as f:
            f.seek(begin + start)
            data = f.read(end - start)
        if len(data) != end - start:
            raise ValueError(f'Truncated vision tensor: {name}')
        digest.update(name.encode() + b'\0' + json.dumps(info, sort_keys=True).encode() + b'\0' + data)
        x = np.frombuffer(data, dtype=DTYPES[info['dtype']]).reshape(info['shape'])
        if info['dtype'] == 'BF16':
            x = (x.astype(np.uint32) << 16).view(np.float32)
        tensors['model.visual.' + name] = x
    metadata = {'tensorfold_converter': VERSION, 'source_vision_sha256': digest.hexdigest(),
                'config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'dtype': 'F16'}
    if output.exists():
        with safe_open(str(output), framework='np') as f:
            if f.metadata() != metadata:
                raise ValueError('Vision cache belongs to another checkpoint or converter')
        return output
    print(f'[sparktalk] converting {len(tensors)} original EXL3 vision tensors', flush=True)
    converted = convert_tensors(tensors, raw['vision_config'])
    temp = output.with_name(output.name + '.tmp-' + str(os.getpid()))
    try:
        save_file(converted, str(temp), metadata=metadata)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    print('[sparktalk] EXL3 vision cache prepared', flush=True)
    return output


if __name__ == '__main__':
    import sys
    print(prepare(sys.argv[1], sys.argv[2]), flush=True)
