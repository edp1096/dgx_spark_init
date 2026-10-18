"""Audit the entire pinned GGUF pair without storing two full checkpoints.

Equal LFS files or matching Xet content trees plus exact boundary bytes prove
unchanged tensors. All changed candidate tensors are compared in full. Any
change outside the supported projection families prevents conversion.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import threading
import time

import requests

from download import atomic, digest

sys.path.append(str(Path(__file__).resolve().parents[1] / 'qwen38_lil_huihui_bf16'))
from xet_audit import Remote

PINS = {'original': '766911a6b7369840a91dbcd95f9f997acaab6cd6',
        'huihui': '43b4c1a8871afcc27e37bf996a431528b3d957e5'}
PREFIXES = {'original': 'Q8_0/', 'huihui': 'unsloth-Q8_0/'}
ALLOWED = re.compile(r'blk\.(\d+)\.(ssm_out|attn_output|ffn_down_shexp)\.weight')


class Incomplete(Exception):
    pass


class Reader:
    def __init__(self, raw):
        self.raw, self.pos = raw, 0

    def take(self, size):
        if size < 0 or self.pos + size > len(self.raw):
            raise Incomplete()
        data = self.raw[self.pos:self.pos + size]
        self.pos += size
        return data

    def number(self, kind):
        return struct.unpack('<' + kind, self.take(struct.calcsize('<' + kind)))[0]

    def string(self):
        return self.take(self.number('Q')).decode('utf-8')

    def value(self, kind):
        primitive = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                     6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if kind in primitive:
            return self.number(primitive[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            subtype, count = self.number('I'), self.number('Q')
            if count > 10_000_000:
                raise ValueError('Unbounded GGUF metadata array')
            if subtype in primitive:
                self.take(count * struct.calcsize('<' + primitive[subtype]))
            else:
                for _ in range(count):
                    self.value(subtype)
            return None
        raise ValueError('Unsupported GGUF metadata type')


def tensor_bytes(shape, kind):
    count = math.prod(shape)
    if kind == 8 and count % 32 == 0:
        return count // 32 * 34
    if kind in (0, 1, 30):
        return count * (4 if kind == 0 else 2)
    raise ValueError('Unsupported GGUF tensor type/geometry')


def parse_header(raw):
    r = Reader(raw)
    if r.take(4) != b'GGUF' or r.number('I') != 3:
        raise ValueError('Expected GGUF v3')
    count, metadata_count = r.number('Q'), r.number('Q')
    if count > 100_000 or metadata_count > 100_000:
        raise ValueError('Unbounded GGUF header')
    metadata, metadata_hashes = {}, {}
    for _ in range(metadata_count):
        begin = r.pos
        name = r.string()
        if name in metadata:
            raise ValueError('Duplicate GGUF metadata key')
        metadata[name] = r.value(r.number('I'))
        metadata_hashes[name] = hashlib.sha256(raw[begin:r.pos]).hexdigest()
    tensors = {}
    for _ in range(count):
        name, ndim = r.string(), r.number('I')
        if name in tensors or not 1 <= ndim <= 4:
            raise ValueError('Invalid GGUF tensor descriptor')
        shape = [r.number('Q') for _ in range(ndim)]
        if any(n <= 0 for n in shape):
            raise ValueError('Empty GGUF tensor dimension')
        kind, offset = r.number('I'), r.number('Q')
        size = tensor_bytes(shape, kind)
        tensors[name] = {'shape': shape, 'type': kind, 'bytes': size,
                         'data_offsets': [offset, offset + size]}
    alignment = metadata.get('general.alignment', 32)
    if alignment <= 0 or alignment > 4096 or alignment & (alignment - 1):
        raise ValueError('Invalid GGUF alignment')
    return {'header_bytes': r.pos, 'start': (r.pos + alignment - 1) // alignment * alignment,
            'tensors': tensors, 'alignment': alignment,
            'metadata_hashes': metadata_hashes,
            'metadata': {k: v for k, v in metadata.items() if not k.startswith('tokenizer.')}}


def complements(length, intervals):
    cursor, result = 0, []
    for start, end in sorted(intervals):
        if not cursor <= start <= end <= length:
            raise ValueError('Overlapping proof intervals')
        if start > cursor:
            result.append((cursor, start))
        cursor = end
    if cursor < length:
        result.append((cursor, length))
    return result


class GGUFRemote(Remote):
    def __init__(self, root, side):
        super().__init__(root, side)
        if self.rev != PINS[side]:
            raise ValueError('Unexpected GGUF revision')
        self.info = {}
        self.network = 0
        self.lock = threading.Lock()

    def bytes(self, name, start, end):
        if start == end:
            return b''
        if not 0 <= start < end <= self.files[name]['size'] or end - start > 32 << 20:
            raise ValueError('Unbounded GGUF byte request')
        key = hashlib.sha256(f'{name}/{start}/{end}'.encode()).hexdigest()
        path = self.cache / (key + '.range')
        if path.exists():
            value = path.read_bytes()
            if len(value) != end - start:
                raise ValueError('Corrupt cached GGUF range')
            return value
        url = self.url(name) + f'?tensor_range={start}-{end}'
        response = self.get(url, headers={'Range': f'bytes={start}-{end-1}'}, stream=True)
        with response:
            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end-1}/{self.files[name]["size"]}':
                raise ValueError('Exact GGUF range not honored')
            value = response.raw.read(end - start + 1)
        if len(value) != end - start:
            raise ValueError('GGUF range length mismatch')
        with self.lock:
            self.network += len(value)
        # Only boundaries/headers are cached here. Selected complete tensors
        # have their own independently hashed files and are not duplicated.
        if end - start <= 2 << 20:
            temp = path.with_suffix('.tmp')
            temp.write_bytes(value)
            temp.replace(path)
        return value

    def file(self, name):
        if name in self.info:
            return self.info[name]
        record = self.files[name]
        key = hashlib.sha256(name.encode()).hexdigest()
        saved = self.cache / (key + '.header.json')
        if saved.exists():
            info = json.loads(saved.read_text())
            if info['revision'] != self.rev or info['lfs_sha256'] != record['lfs']['sha256']:
                raise ValueError('Cached header belongs to a different source')
            self.info[name] = info
            return info
        with self.get(self.url(name), allow_redirects=False) as response:
            xet_hash = response.headers.get('X-Xet-Hash', '')
            linked = response.headers.get('X-Linked-Etag', '').strip('"')
        if linked != record['lfs']['sha256'] or len(xet_hash) != 64:
            raise ValueError('Resolved GGUF identity differs from pinned metadata')
        for limit in [65536, 2 << 20, 16 << 20]:
            raw = self.bytes(name, 0, min(limit, record['size']))
            try:
                info = parse_header(raw)
                break
            except Incomplete:
                continue
        else:
            raise ValueError('GGUF header exceeds 16 MiB')
        cursor = 0
        for tensor in sorted(info['tensors'].values(), key=lambda t: t['data_offsets'][0]):
            lo, hi = tensor['data_offsets']
            aligned = (cursor + info['alignment'] - 1) // info['alignment'] * info['alignment']
            if lo != aligned or hi + info['start'] > record['size']:
                raise ValueError('GGUF tensor overlap/gap/out-of-bounds')
            cursor = hi
        if info['start'] + cursor != record['size']:
            raise ValueError('Unindexed trailing GGUF bytes')
        info.update(repo=self.repo, revision=self.rev, file=name, xet_hash=xet_hash,
                    lfs_sha256=linked, size=record['size'],
                    header_sha256=hashlib.sha256(raw[:info['header_bytes']]).hexdigest())
        atomic(saved, info)
        self.info[name] = info
        return info

    def stream_hash(self, name, start, end):
        result = hashlib.sha256()
        for pos in range(start, end, 32 << 20):
            result.update(self.bytes(name, pos, min(end, pos + (32 << 20))))
        return result.hexdigest()


def selected(root, a, b, name, ai, bi):
    size = ai['tensors'][name]['bytes']
    hashes = {}
    for side, remote, info in [('original', a, ai), ('huihui', b, bi)]:
        path = root / 'ranges' / side / (name + '.q8')
        path.parent.mkdir(parents=True, exist_ok=True)
        start = info['start'] + info['tensors'][name]['data_offsets'][0]
        if not path.exists():
            temp = path.with_suffix('.partial')
            done = temp.stat().st_size if temp.exists() else 0
            if done > size:
                raise ValueError('Selected partial exceeds tensor length')
            with temp.open('ab') as out:
                while done < size:
                    end = min(size, done + (32 << 20))
                    out.write(remote.bytes(info['file'], start + done, start + end))
                    done = end
            temp.replace(path)
        if path.stat().st_size != size:
            raise ValueError('Selected tensor length mismatch')
        hashes[side] = digest(path)
    return {'name': name, 'shape': ai['tensors'][name]['shape'], 'type': 8,
            'bytes': size, 'changed': hashes['original'] != hashes['huihui'],
            'proof': 'complete_tensor_sha256',
            **{side + '_sha256': h for side, h in hashes.items()}}


def compare(root, a, b, name, ai, bi):
    receipt = root / 'audit' / 'receipts' / (name + '.json')
    if receipt.exists():
        row = json.loads(receipt.read_text())
        if row.get('proof') == 'complete_tensor_sha256':
            for side in ('original', 'huihui'):
                if digest(root / 'ranges' / side / (name + '.q8')) != row[side + '_sha256']:
                    raise ValueError('Previously verified selected tensor changed')
        return row
    x, y = ai['tensors'][name], bi['tensors'][name]
    if any(x[k] != y[k] for k in ['shape', 'type', 'bytes']):
        raise ValueError('GGUF tensor layout differs: ' + name)
    match = ALLOWED.fullmatch(name)
    if match:
        if not 0 <= int(match[1]) < 48 or x['type'] != 8:
            raise ValueError('Unsupported projection geometry')
        row = selected(root, a, b, name, ai, bi)
    elif ai['lfs_sha256'] == bi['lfs_sha256']:
        row = {'name': name, 'bytes': x['bytes'], 'changed': False,
               'proof': 'identical_pinned_lfs_file', 'file_sha256': ai['lfs_sha256']}
    else:
        ga, gb = a.gap(ai, name), b.gap(bi, name)
        source_nodes, pos = {}, ga['gap_start'] - ga['start']
        for node in ga['nodes']:
            source_nodes[(pos, node['size'])] = node['hash']
            pos += node['size']
        covered, pos = [], gb['gap_start'] - gb['start']
        for node in gb['nodes']:
            if source_nodes.get((pos, node['size'])) == node['hash']:
                covered.append((pos, pos + node['size']))
            pos += node['size']
        missing = complements(x['bytes'], covered)
        row = {'name': name, 'bytes': x['bytes'], 'changed': False,
               'proof': 'matching_xet_subtrees_and_exact_complement',
               'covered_intervals': covered, 'complement': []}
        if sum(hi - lo for lo, hi in missing) > 4 << 20:
            print('STREAM VERIFY', name, x['bytes'], flush=True)
            ah = a.stream_hash(ai['file'], ga['start'], ga['end'])
            bh = b.stream_hash(bi['file'], gb['start'], gb['end'])
            row.update(proof='complete_streamed_sha256', changed=ah != bh,
                       original_sha256=ah, huihui_sha256=bh)
        else:
            for lo, hi in missing:
                av = a.bytes(ai['file'], ga['start'] + lo, ga['start'] + hi)
                bv = b.bytes(bi['file'], gb['start'] + lo, gb['start'] + hi)
                row['changed'] = row['changed'] or av != bv
                row['complement'].append({'start': lo, 'end': hi,
                    'original_sha256': hashlib.sha256(av).hexdigest(),
                    'huihui_sha256': hashlib.sha256(bv).hexdigest()})
    atomic(receipt, row)
    if row['changed']:
        print('CHANGED', name, row['bytes'], flush=True)
    return row


def compare_vision(root, a, b):
    receipt = root / 'audit/vision.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    ai = a.file('mmproj-BF16.gguf')
    bi = b.file('mmproj-Qwen3.8-Flash-Next-BF16.gguf')
    if ai['tensors'] != bi['tensors'] or ai['size'] - ai['start'] != bi['size'] - bi['start']:
        raise ValueError('Vision tensor inventories/layouts differ')
    keys = ai['metadata_hashes'].keys() | bi['metadata_hashes'].keys()
    different = [k for k in keys if ai['metadata_hashes'].get(k) != bi['metadata_hashes'].get(k)]
    descriptive = {'general.name', 'general.basename', 'general.finetune', 'general.description',
        'general.author', 'general.url', 'general.source.url', 'general.quantized_by',
        'general.repo_url', 'general.tags'}
    if set(different) - descriptive:
        raise ValueError('Functional vision metadata differs: ' + ', '.join(different))
    print('VERIFY VISION PAYLOADS', ai['size'] - ai['start'], flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(a.stream_hash, ai['file'], ai['start'], ai['size'])
        right = pool.submit(b.stream_hash, bi['file'], bi['start'], bi['size'])
        ah, bh = left.result(), right.result()
    if ah != bh:
        raise ValueError('Vision source weights changed; do not omit them from transfer')
    row = {'status': 'equal', 'tensor_count': len(ai['tensors']),
        'original_file': ai['file'], 'huihui_file': bi['file'],
        'original_file_sha256': ai['lfs_sha256'], 'huihui_file_sha256': bi['lfs_sha256'],
        'payload_bytes': ai['size'] - ai['start'], 'payload_sha256': ah,
        'metadata_differences': sorted(different), 'proof': 'complete_streamed_payload_sha256'}
    atomic(receipt, row)
    return row


def run(root, workers):
    previous = root / 'audit/tensor-audit.json'
    previous_network = json.loads(previous.read_text()).get('network_bytes', 0) if previous.exists() else 0
    a, b = GGUFRemote(root, 'original'), GGUFRemote(root, 'huihui')
    files = [{Path(n).name: n for n in r.files if n.startswith(PREFIXES[r.side])}
             for r in (a, b)]
    if files[0].keys() != files[1].keys() or len(files[0]) != 6:
        raise ValueError('Unexpected GGUF shard inventory')
    tasks, inventory, file_rows = [], set(), []
    for filename in sorted(files[0]):
        ai, bi = a.file(files[0][filename]), b.file(files[1][filename])
        if ai['tensors'].keys() != bi['tensors'].keys():
            raise ValueError('GGUF tensor inventories differ')
        file_rows.append({'file': filename, 'original': ai, 'huihui': bi})
        for name in ai['tensors']:
            if name in inventory:
                raise ValueError('Duplicated tensor across GGUF shards')
            inventory.add(name)
            tasks.append((name, ai, bi))
    atomic(root / 'audit' / 'files.json', file_rows)
    rows, started = [], time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(lambda task: compare(root, a, b, *task), tasks):
            rows.append(row)
            if len(rows) % 20 == 0:
                atomic(root / 'audit-progress.json', {'checked': len(rows), 'total': len(tasks),
                    'changed': sum(r['changed'] for r in rows), 'network_bytes': a.network + b.network,
                    'seconds': time.monotonic() - started})
    changed = [row for row in rows if row['changed']]
    unexpected = [row['name'] for row in changed if not ALLOWED.fullmatch(row['name'])]
    vision = compare_vision(root, a, b)
    report = {'status': 'complete' if not unexpected else 'unexpected_changes',
              'sources': PINS, 'tensor_count': len(rows), 'changed_count': len(changed),
              'unexpected_changes': unexpected, 'tensors': rows, 'vision': vision,
              'network_bytes': previous_network + a.network + b.network, 'seconds': time.monotonic() - started,
              'audit_program_sha256': digest(Path(__file__)),
              'input_files': [{'file': f['file'],
                  **{side: {k: f[side][k] for k in ['repo', 'revision', 'file', 'size', 'lfs_sha256', 'xet_hash', 'header_sha256']}
                     for side in ['original', 'huihui']}} for f in file_rows]}
    atomic(root / 'audit' / 'tensor-audit.json', report)
    if unexpected:
        raise ValueError('Changes outside approved projection families: ' + ', '.join(unexpected))
    if len(rows) != 1224 or not changed:
        raise ValueError('Incomplete or empty source audit')
    print('SOURCE AUDIT COMPLETE', len(rows), 'tensors;', len(changed), 'changed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    try:
        run(args.root, args.workers)
    except requests.RequestException as exc:
        raise SystemExit('Audit transport failed: ' + type(exc).__name__) from None
