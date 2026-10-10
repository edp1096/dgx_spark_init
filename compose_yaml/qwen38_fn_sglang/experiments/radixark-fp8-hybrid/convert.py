"""Create a separate, verified FP8 side-layer view of a cached NVFP4 model.

Uses blazux's 128x128 e4m3 quantization formula. Original files are read only;
unchanged shards are relative symlinks. No network access or GPU is needed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TARGET = re.compile(
    r"^model\.language_model\.layers\.\d+\.("
    r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.shared_expert\.(gate_proj|up_proj|down_proj))\.weight$"
)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def convert(source, destination):
    source = source.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    targets = {k: v for k, v in index['weight_map'].items() if TARGET.fullmatch(k)}
    assert len(targets) == 300, f'Unexpected checkpoint: {len(targets)} target tensors'
    files = sorted(set(targets.values()))
    manifest = dict(source=str(source), algorithm='block128-e4m3-dynamic',
                    upstream='blazux/qwen3.8-Flash-DGX@779185b3615211952417f47231385000a0866ec3',
                    source_hashes={}, converted={}, tensors={}, before_bytes=0, after_bytes=0)
    # Copy configuration rather than editing a link into the original snapshot.
    for path in source.iterdir():
        if not path.is_file() or path.name in files or path.name == 'model.safetensors.index.json':
            continue
        if path.suffix == '.json':
            (destination / path.name).write_bytes(path.read_bytes())
        else:
            (destination / path.name).symlink_to(os.path.relpath(path, destination))
    torch.set_num_threads(8)
    for filename in files:
        path = source / filename
        manifest['source_hashes'][filename] = sha(path)
        with safe_open(path, framework='pt', device='cpu') as handle:
            metadata = handle.metadata()
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
            unchanged = {k: v for k, v in tensors.items() if k not in targets}
            for name in sorted(set(tensors) & targets.keys()):
                weight = tensors.pop(name)
                assert weight.dtype == torch.bfloat16 and weight.ndim == 2, name
                rows, cols = weight.shape
                assert rows % 128 == cols % 128 == 0, (name, weight.shape)
                wf = weight.float().reshape(rows // 128, 128, cols // 128, 128)
                scale = wf.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / 448.0
                quant = (wf / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                restored = quant.float() * scale
                relative_l2 = ((restored - wf).norm() / wf.norm().clamp_min(1e-12)).item()
                max_relative = ((restored - wf).abs().max() / wf.abs().max().clamp_min(1e-12)).item()
                assert torch.isfinite(restored).all() and relative_l2 < 0.04 and max_relative < 0.1, name
                tensors[name] = quant.reshape(rows, cols).contiguous()
                scale_name = name[:-len('.weight')] + '.weight_scale_inv'
                tensors[scale_name] = scale.squeeze(1).squeeze(-1).contiguous()
                index['weight_map'][scale_name] = filename
                before = weight.numel() * weight.element_size()
                after = tensors[name].numel() + tensors[scale_name].numel() * 4
                manifest['before_bytes'] += before
                manifest['after_bytes'] += after
                manifest['tensors'][name] = dict(shape=list(weight.shape), relative_l2=relative_l2,
                                                max_relative=max_relative)
            temporary = destination / (filename + '.tmp')
            save_file(tensors, str(temporary), metadata=metadata)
            temporary.replace(destination / filename)
            # Check every unaffected tensor inside rewritten shards, including MTP.
            with safe_open(destination / filename, framework='pt', device='cpu') as verify:
                assert set(verify.keys()) == set(tensors)
                for name, tensor in unchanged.items():
                    assert torch.equal(verify.get_tensor(name).reshape(-1).view(torch.uint8),
                                       tensor.reshape(-1).view(torch.uint8)), name
                for name in set(targets) & tensors.keys():
                    assert verify.get_tensor(name).dtype == torch.float8_e4m3fn, name
            assert sha(path) == manifest['source_hashes'][filename], 'Original shard changed'
        manifest['converted'][filename] = dict(sha256=sha(destination / filename),
                                               bytes=(destination / filename).stat().st_size)
        print(json.dumps(dict(shard=filename, converted=True, seconds=time.monotonic()-started)), flush=True)
        del tensors, unchanged
    total_size = 0
    for filename in set(index['weight_map'].values()):
        with (destination / filename).open('rb') as handle:
            size = struct.unpack('<Q', handle.read(8))[0]
            header = json.loads(handle.read(size))
        total_size += sum(v['data_offsets'][1] - v['data_offsets'][0]
                          for k, v in header.items() if k != '__metadata__')
    index.setdefault('metadata', {})['total_size'] = total_size
    (destination / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2))
    manifest['seconds'] = time.monotonic() - started
    manifest['saved_bytes'] = manifest['before_bytes'] - manifest['after_bytes']
    (destination / 'fp8-hybrid-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(done=True, tensors=len(manifest['tensors']), saved_bytes=manifest['saved_bytes'],
                         seconds=manifest['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
