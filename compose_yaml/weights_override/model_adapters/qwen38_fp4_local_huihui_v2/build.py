"""Transfer audited Huihui v2 Q8 deltas into FP4-Local MXFP8 projections.

The NVFP4/CSF expert tensors are immutable. An independent checkpoint is
published locally only after complete source, numerical and byte-range checks.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import torch
from modelopt.torch.quantization.qtensor.mxfp8_tensor import MXFP8QTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from weights_core.safetensors_io import Checkpoint, read_header
from download import atomic, digest

BASE_REV = 'b9ce18a6887e18c089fdea8509a33e63429a3300'
PINS = {'original': '766911a6b7369840a91dbcd95f9f997acaab6cd6',
        'huihui': '43b4c1a8871afcc27e37bf996a431528b3d957e5'}
SUFFIX = {'ssm_out': 'linear_attn.out_proj', 'attn_output': 'self_attn.o_proj',
          'ffn_down_shexp': 'mlp.shared_expert.down_proj'}
PATTERN = re.compile(r'blk\.(\d+)\.(ssm_out|attn_output|ffn_down_shexp)\.weight')


def mapped(row):
    match = PATTERN.fullmatch(row['name'])
    if not match or not 0 <= int(match[1]) < 48 or row['type'] != 8:
        raise ValueError('Unsupported changed tensor: ' + row['name'])
    shape = list(reversed(row['shape']))
    expected = [2560, 640 if match[2] == 'ffn_down_shexp' else 6144]
    if shape != expected or row['bytes'] != math.prod(shape) // 32 * 34:
        raise ValueError('Unexpected projection dimensions')
    return f'model.language_model.layers.{match[1]}.{SUFFIX[match[2]]}', shape, match[2]


class Projections(Checkpoint):
    """Read selected, fully downloaded shards while other shards download."""
    def __init__(self, root, prefixes):
        self.root = root.resolve(strict=True)
        self.tensors = {}
        index = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        wanted = {prefix + suffix for prefix in prefixes for suffix in ('.weight', '.weight_scale')}
        shards = {index[key] for key in wanted}
        for shard in shards:
            if '\\' in shard or ':' in shard or any(p in ('', '.', '..') for p in shard.split('/')):
                raise ValueError('Unsafe shard path')
            header, start = read_header(self.root / shard)
            for name in wanted:
                if index[name] == shard:
                    self.tensors[name] = {**header[name], 'shard': shard, 'start': start}
        if self.tensors.keys() != wanted:
            raise ValueError('Incomplete projection subset')


def q8_decode(raw, rows, columns):
    if columns % 32 or len(raw) != rows * columns // 32 * 34:
        raise ValueError('Q8_0 payload length mismatch')
    blocks = np.frombuffer(raw, dtype=np.dtype([('d', '<f2'), ('q', 'i1', (32,))]))
    values = blocks['d'].astype(np.float32)[:, None] * blocks['q'].astype(np.float32)
    out = torch.from_numpy(values.reshape(rows, columns).copy())
    if not torch.isfinite(out).all():
        raise ValueError('Nonfinite source Q8 values')
    return out


def restore_gdn(value, config):
    c = config['text_config']
    k, v, d = c['linear_num_key_heads'], c['linear_num_value_heads'], c['linear_value_head_dim']
    if v % k or value.shape[1] != v * d:
        raise ValueError('GDN column geometry mismatch')
    return value.reshape(value.shape[0], v // k, k, d).transpose(1, 2).contiguous().reshape(value.shape)


def read_rows(cp, name, first, rows):
    t = cp.tensors[name]
    if len(t['shape']) != 2 or t['dtype'] not in ('F8_E4M3', 'U8'):
        raise ValueError('Expected an MXFP8 matrix or UE8M0 scales')
    columns = t['shape'][1]
    with (cp.root / t['shard']).open('rb') as f:
        f.seek(t['start'] + t['data_offsets'][0] + first * columns)
        data = f.read(rows * columns)
    if len(data) != rows * columns:
        raise ValueError('Truncated MXFP8 matrix')
    kind = torch.float8_e4m3fn if t['dtype'] == 'F8_E4M3' else torch.uint8
    return torch.frombuffer(bytearray(data), dtype=kind).reshape(rows, columns)


def blob(value):
    return value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def transform(original, donor, weights, scales, config, gdn=False):
    rows, columns = weights.shape
    if weights.dtype != torch.float8_e4m3fn or scales.dtype != torch.uint8 or list(scales.shape) != [rows, columns // 32]:
        raise ValueError('Unexpected MXFP8 encoding')
    raw_original = original
    if gdn:
        original, donor = restore_gdn(original, config), restore_gdn(donor, config)
    base = (weights.float().reshape(rows, -1, 32) * torch.exp2(scales.float() - 127)[..., None]).reshape(rows, columns)
    delta = donor - original
    target = base + delta
    if not torch.isfinite(target).all():
        raise ValueError('Nonfinite conversion target')
    packed, new_scales = MXFP8QTensor.quantize(target.to(torch.bfloat16))
    restored = packed.dequantize(dtype=torch.float32, scale=new_scales)
    if not torch.isfinite(restored).all():
        raise ValueError('Nonfinite reconstructed result')
    effect = restored - base
    norm = lambda x: float(torch.sum(x.double().square()))
    metrics = {'base_norm2': norm(base), 'target_norm2': norm(target),
               'delta_norm2': norm(delta), 'error_norm2': norm(target - restored),
               'effect_norm2': norm(effect), 'source_alignment_error2': norm(base - original),
               'unpermuted_alignment_error2': norm(base - raw_original),
               'delta_dot_effect': float(torch.sum(delta.double() * effect.double()))}
    return packed._quantized_data, new_scales, metrics


def complement_hash(path, intervals):
    result, cursor = hashlib.sha256(), 0
    with path.open('rb') as f:
        for lo, hi in sorted(intervals) + [(path.stat().st_size, path.stat().st_size)]:
            if not cursor <= lo <= hi <= path.stat().st_size:
                raise ValueError('Overlapping patch intervals')
            f.seek(cursor)
            left = lo - cursor
            while left:
                data = f.read(min(left, 4 << 20))
                if not data:
                    raise ValueError('Truncated complement')
                result.update(data)
                left -= len(data)
            cursor = hi
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return result.hexdigest()


def process(root, cp, config, row, output=None):
    prefix, shape, kind = mapped(row)
    for side in ('original', 'huihui'):
        path = root / 'ranges' / side / (row['name'] + '.q8')
        if digest(path) != row[side + '_sha256']:
            raise ValueError('Source range changed after audit')
    weight, scale = cp.tensors[prefix + '.weight'], cp.tensors[prefix + '.weight_scale']
    if weight['shape'] != shape or weight['dtype'] != 'F8_E4M3' or scale['dtype'] != 'U8' or scale['shape'] != [shape[0], shape[1] // 32]:
        raise ValueError('Source/base projection mapping mismatch')
    stats = {}
    qrow = shape[1] // 32 * 34
    with (root / 'ranges/original' / (row['name'] + '.q8')).open('rb') as a, (root / 'ranges/huihui' / (row['name'] + '.q8')).open('rb') as b:
        for first in range(0, shape[0], 128):
            rows = min(128, shape[0] - first)
            original, donor = [q8_decode(f.read(rows * qrow), rows, shape[1]) for f in (a, b)]
            w = read_rows(cp, prefix + '.weight', first, rows)
            s = read_rows(cp, prefix + '.weight_scale', first, rows)
            qw, qs, metrics = transform(original, donor, w, s, config, kind == 'ssm_out')
            for key, value in metrics.items():
                stats[key] = stats.get(key, 0.) + value
            if output is not None:
                for descriptor, value in [(weight, qw), (scale, qs)]:
                    raw = blob(value)
                    if value.shape[1] != descriptor['shape'][1] or len(raw) != rows * descriptor['shape'][1]:
                        raise ValueError('Quantizer changed stored layout')
                    with (output / descriptor['shard']).open('r+b') as out:
                        out.seek(descriptor['start'] + descriptor['data_offsets'][0] + first * descriptor['shape'][1])
                        out.write(raw)
    stats['target_relative_error'] = math.sqrt(stats['error_norm2'] / max(stats['target_norm2'], 1e-30))
    stats['source_alignment_relative_error'] = math.sqrt(stats['source_alignment_error2'] / max(stats['base_norm2'], 1e-30))
    stats['unpermuted_alignment_relative_error'] = math.sqrt(stats['unpermuted_alignment_error2'] / max(stats['base_norm2'], 1e-30))
    stats['delta_relative_l2'] = math.sqrt(stats['delta_norm2'] / max(stats['base_norm2'], 1e-30))
    stats['delta_effective_cosine'] = stats['delta_dot_effect'] / max(math.sqrt(stats['delta_norm2'] * stats['effect_norm2']), 1e-30)
    if (not all(math.isfinite(v) for v in stats.values()) or stats['target_relative_error'] > .05
            or stats['source_alignment_relative_error'] > (.5 if kind == 'ffn_down_shexp' else .08)
            or stats['delta_effective_cosine'] <= 0):
        raise ValueError('Numerical/alignment validation failed: ' + row['name'] + ' ' + json.dumps(stats))
    return {'name': row['name'], 'destination': prefix, 'shape': shape, **stats}


def run(root, output, probe=False, owner=None):
    torch.set_num_threads(4)
    base = root / 'base'
    config = json.loads((base / 'config.json').read_text())
    quantizer = {'library': 'NVIDIA ModelOpt', 'version': importlib.metadata.version('nvidia-modelopt'),
                 'torch': torch.__version__,
                 'implementation_sha256': digest(Path(inspect.getsourcefile(MXFP8QTensor)))}
    if config['quantization_config']['quant_algo'] != 'MIXED_PRECISION' or config['text_config']['ple_embedding_dtype'] != 'nvfp4':
        raise ValueError('Unexpected FP4-Local configuration')
    if probe:
        names = ['blk.0.ssm_out.weight', 'blk.3.attn_output.weight', 'blk.0.ffn_down_shexp.weight']
        rows = [json.loads((root / 'audit/receipts' / (name + '.json')).read_text()) for name in names]
        cp = Projections(base, [mapped(r)[0] for r in rows])
        result = [process(root, cp, config, row) for row in rows]
        atomic(root / 'reports/numeric-probe.json', {'status': 'passed', 'quantizer': quantizer, 'tensors': result})
        for row in result:
            print(row['name'], 'alignment', row['source_alignment_relative_error'], 'requant', row['target_relative_error'], 'cosine', row['delta_effective_cosine'], flush=True)
        return
    audit = json.loads((root / 'audit/tensor-audit.json').read_text())
    receipt = json.loads((base / 'download-receipt.json').read_text())
    if (audit['status'] != 'complete' or audit['sources'] != PINS or audit['tensor_count'] != 1224
            or audit['unexpected_changes'] or audit.get('vision', {}).get('status') != 'equal'):
        raise ValueError('A complete pinned source audit is required')
    if receipt['status'] != 'verified' or receipt['revision'] != BASE_REV:
        raise ValueError('A verified pinned FP4-Local checkpoint is required')
    rows = [r for r in audit['tensors'] if r['changed']]
    if not rows or any(not PATTERN.fullmatch(r['name']) for r in rows):
        raise ValueError('Unsupported audit targets')
    output, base = output.resolve(), base.resolve()
    if output == base or output in base.parents or base in output.parents:
        raise ValueError('Source and output must be independent')
    work = output.with_name('.' + output.name + '.partial')
    if output.exists() or work.exists():
        raise FileExistsError('Refusing to replace an existing candidate')
    cp = Checkpoint(base)
    intervals = {}
    targets = set()
    for row in rows:
        prefix, shape, _ = mapped(row)
        for suffix, dtype, dimensions in [('.weight', 'F8_E4M3', shape), ('.weight_scale', 'U8', [shape[0], shape[1] // 32])]:
            name = prefix + suffix
            t = cp.tensors[name]
            if t['dtype'] != dtype or t['shape'] != dimensions:
                raise ValueError('Base layout differs from approved MXFP8 target')
            intervals.setdefault(t['shard'], []).append(tuple(t['start'] + x for x in t['data_offsets']))
            targets.add(name)
    if any('.experts.' in name or 'csf' in name for name in targets):
        raise ValueError('Routed NVFP4/CSF tensors must remain immutable')
    total = sum(f['size'] for f in receipt['files'])
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < total + (8 << 30):
        raise ValueError('Insufficient space for independent output')
    work.mkdir()
    manifest = {'status': 'building', 'runtime_validated': False,
        'base': receipt['repo'], 'base_revision': BASE_REV, 'source_revisions': PINS,
        'method': 'MXFP8(BF16(DQ(FP4-Local) + restore_HF(DQ(Huihui v2 Q8) - DQ(Unsloth Q8))))',
        'activation_scales': 'unchanged; no new calibration',
        'source_audit_sha256': digest(root / 'audit/tensor-audit.json'),
        'converter_sha256': digest(Path(__file__)), 'started': time.time(),
        'quantizer': quantizer,
        'source_shards': {}, 'output_shards': {}, 'complement_hashes': {},
        'changed_tensor_names': sorted(targets), 'tensors': []}
    def save():
        atomic(work / 'transfer-manifest.json', manifest)
    save()
    for f in receipt['files']:
        name = f['rfilename']
        if name == 'README.md':
            continue
        src, dest = base / name, work / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with src.open('rb') as fi, dest.open('xb') as fo:
            for data in iter(lambda: fi.read(4 << 20), b''):
                h.update(data)
                fo.write(data)
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(fi.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if h.hexdigest() != f['verified_sha256'] or os.path.samefile(src, dest):
            raise ValueError('Pinned source changed or output is aliased')
        if name.endswith('.safetensors'):
            manifest['source_shards'][name] = h.hexdigest()
            print('COPIED', name, flush=True)
        save()
    for row in rows:
        result = process(root, cp, config, row, work)
        manifest['tensors'].append(result)
        save()
        print('PATCHED', row['name'], 'error', result['target_relative_error'], flush=True)
    candidate = Checkpoint(work)
    if candidate.tensors != cp.tensors:
        raise ValueError('Candidate index/header layout changed')
    for name, source_hash in manifest['source_shards'].items():
        original = digest(base / name)
        actual = digest(work / name)
        if original != source_hash:
            raise ValueError('Source checkpoint mutated')
        if name in intervals:
            left, right = [complement_hash(directory / name, intervals[name]) for directory in (base, work)]
            if left != right:
                raise ValueError('Untargeted checkpoint bytes changed')
            manifest['complement_hashes'][name] = left
        elif actual != source_hash:
            raise ValueError('Untouched shard changed')
        manifest['output_shards'][name] = actual
    for f in receipt['files']:
        name = f['rfilename']
        if name != 'README.md' and not name.endswith('.safetensors') and digest(work / name) != f['verified_sha256']:
            raise ValueError('Runtime metadata changed')
    shutil.copyfile(root / 'audit/tensor-audit.json', work / 'source-audit.json')
    (work / 'README.md').write_text('''---
base_model:
- nvidia/Qwen3.8-Flash-Next-FP4-Local
- huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF
pipeline_tag: image-text-to-text
library_name: vllm
license: other
license_name: nvidia-open-model-agreement
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
tags: [nvfp4, mxfp8, modelopt, abliterated, qwen3.8]
---

# Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local

NVIDIA FP4-Local with the audited Huihui v2 changes transferred from matched Unsloth/Huihui Q8_0 weights. Changed projections are reconstructed, updated, rounded to BF16 and encoded as MXFP8. Routed NVFP4 experts and their CSF scales, PLE, MTP, vision and tokenizer retain the NVIDIA checkpoint bytes.

See `transfer-manifest.json` for pinned sources, changed tensors and quantization errors, and `source-audit.json` for the complete source comparison. Runtime results are recorded separately in `runtime-qualification.json` when available.

Sources: [NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), [Huihui v2](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF), [Unsloth Q8](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF). The NVIDIA Open Model Agreement and the base model's [Qwen Community License](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) apply.
''')
    manifest.update(status='candidate_verified', finished=time.time(), changed_projections=len(rows))
    save()
    if owner:
        uid, gid = (int(part) for part in owner.split(':'))
        for path in list(work.rglob('*')) + [work]:
            os.chown(path, uid, gid, follow_symlinks=False)
    work.rename(output)
    print('CANDIDATE VERIFIED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--owner', help='UID:GID for the completed independent artifact')
    args = parser.parse_args()
    if not args.probe and args.output is None:
        parser.error('--output is required for conversion')
    run(args.root, args.output, args.probe, args.owner)
