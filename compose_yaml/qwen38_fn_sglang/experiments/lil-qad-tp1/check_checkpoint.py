"""Require every indexed shard before a local-only test boot."""
import json,pathlib,struct,sys
p=pathlib.Path(sys.argv[1])
if not (p/'model.safetensors.index.json').is_file():
    raise SystemExit('Checkpoint incomplete: tensor index missing')
index=json.loads((p/'model.safetensors.index.json').read_text())
missing=[name for name in sorted(set(index['weight_map'].values())) if not (p/name).is_file()]
if missing:raise SystemExit(f'Checkpoint incomplete: {len(missing)} shards missing')
for name in sorted(set(index['weight_map'].values())):
    f=p/name
    with f.open('rb') as stream:
        header_size=struct.unpack('<Q',stream.read(8))[0]
        # The scalar-heavy final shard has a legitimate ~9 MiB JSON header.
        if not 2 <= header_size <= min(64*2**20, f.stat().st_size-8):
            raise SystemExit(f'Invalid header size: {name}')
        header=json.loads(stream.read(header_size))
    expected=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')+8+header_size
    if f.stat().st_size!=expected:raise SystemExit(f'Truncated shard: {name}')
    for key,file in index['weight_map'].items():
        if file==name and key not in header:raise SystemExit(f'Missing tensor: {key}')
print('All checkpoint shards and indexed tensor headers complete')
