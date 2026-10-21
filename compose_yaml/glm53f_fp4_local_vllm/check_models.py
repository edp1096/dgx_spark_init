"""Validate a complete GLM mixed-precision checkpoint before either rank loads it."""
from pathlib import Path
import json, struct, sys

p = Path(sys.argv[1])
config = json.loads((p / 'config.json').read_text())
if config.get('architectures') != ['Glm5NextForConditionalGeneration'] or config.get('quantization_config', {}).get('quant_algo') != 'MIXED_PRECISION':
    raise ValueError('Expected GLM 5.3 Flash FP4-Local mixed-precision checkpoint')
index = json.loads((p / 'model.safetensors.index.json').read_text())['weight_map']
files = set(index.values())
if len(files) != 42:
    raise ValueError('Expected all 42 FP4-Local shards')
for filename in sorted(files):
    if Path(filename).is_absolute() or '..' in Path(filename).parts:
        raise ValueError('Unsafe shard path')
    shard = p / filename
    with shard.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if not 2 <= header_size <= 64 * 2**20:
            raise ValueError('Invalid safetensors header size')
        header = json.loads(stream.read(header_size))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    expected = {k for k, v in index.items() if v == filename}
    if not expected.issubset(tensors):
        raise ValueError('Indexed tensor missing from ' + filename)
    size = shard.stat().st_size - 8 - header_size
    if not tensors or size != max(v['data_offsets'][1] for v in tensors.values()):
        raise ValueError('Incomplete shard: ' + filename)
    for tensor in tensors.values():
        lo, hi = tensor['data_offsets']
        if not 0 <= lo <= hi <= size:
            raise ValueError('Invalid tensor offsets in ' + filename)
for filename in ['tokenizer.json', 'tokenizer_config.json', 'processor_config.json']:
    json.loads((p / filename).read_text())
print('All 42 mixed-precision FP4-Local shard structures and tokenizer/processor files verified.', flush=True)
