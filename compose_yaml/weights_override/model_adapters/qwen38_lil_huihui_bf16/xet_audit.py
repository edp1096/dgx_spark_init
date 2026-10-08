"""Compare pinned BF16 tensors using Xet hash subtrees and boundary bytes.

Only metadata and bounded boundary fragments per side are fetched here.
Uncertified tensors are reported for later range download, never assumed equal.
No large tensor is downloaded by this program. Cached partial shard prefixes
from the cancelled full audit are reused whenever they cover a requested range.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import threading
import time
import requests
from fetch_audit import atomic_json, manifest

MAX_BOUNDARY = 2 << 20
NETWORK_BOUNDARY_BUDGET = 2 << 30
network_lock = threading.Lock()
network_reserved = 0


class Remote:
    def __init__(self, root, side):
        self.root, self.side = root, side
        self.api, self.files = manifest(root, side)
        self.repo, self.rev = self.api['id'], self.api['sha']
        self.token_lock = threading.Lock()
        self.token_time = float('-inf')
        self.thread = threading.local()
        self.cache = root / 'audit' / 'xet' / side
        self.cache.mkdir(parents=True, exist_ok=True)

    def session(self):
        if not hasattr(self.thread, 'session'):
            self.thread.session = requests.Session()
        return self.thread.session

    def get(self, url, **kwargs):
        for attempt in range(5):
            try:
                r = self.session().get(url, timeout=(20, 90), **kwargs)
                r.raise_for_status()
                return r
            except requests.RequestException:
                if attempt == 4: raise
                time.sleep(min(20, 2 ** attempt))

    def auth(self):
        with self.token_lock:
            if time.monotonic() - self.token_time > 300:
                t = self.get(f'https://huggingface.co/api/models/{self.repo}/xet-read-token/{self.rev}').json()
                self.token, self.cas = t['accessToken'], t['casUrl']
                self.token_time = time.monotonic()
            return self.cas, {'Authorization': 'Bearer ' + self.token}

    def url(self, name):
        return f'https://huggingface.co/{self.repo}/resolve/{self.rev}/{name}'

    def bytes(self, name, start, end):
        global network_reserved
        if start == end: return b''
        if not 0 <= start < end <= self.files[name]['size'] or end - start > MAX_BOUNDARY:
            raise ValueError('Refusing an unbounded/bad byte request')
        # These paths belong exclusively to the same pinned earlier audit.
        local = self.root / 'audit' / 'scratch' / self.side / name
        for candidate in [local, local.with_suffix(local.suffix + '.partial')]:
            if candidate.exists() and candidate.stat().st_size >= end:
                with candidate.open('rb') as f:
                    f.seek(start); data = f.read(end - start)
                if len(data) != end - start: raise ValueError('Truncated local prefix')
                return data
        cache = self.cache / f'{name}.{start}-{end}.range'
        if cache.exists():
            data = cache.read_bytes()
            if len(data) != end-start: raise ValueError('Corrupt cached byte range')
            return data
        with network_lock:
            if network_reserved + end-start > NETWORK_BOUNDARY_BUDGET:
                raise ValueError('Boundary audit reached its 2 GiB network budget; refusing further downloads')
            network_reserved += end-start
        r = self.get(self.url(name) + f'?audit_range={start}-{end}', headers={'Range': f'bytes={start}-{end-1}'}, stream=True)
        with r:
            if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end-1}/{self.files[name]["size"]}':
                raise ValueError('Exact range not honored; response body not downloaded')
            data = r.raw.read(end-start+1)
        if len(data) != end-start: raise ValueError('Byte range length mismatch')
        tmp = cache.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(cache)
        return data

    def file(self, name):
        cache = self.cache / (name + '.header.json')
        if cache.exists(): return json.loads(cache.read_text())
        r = self.get(self.url(name), allow_redirects=False)
        xet_hash = r.headers.get('X-Xet-Hash')
        if not xet_hash or len(xet_hash) != 64:
            raise ValueError('Xet file identity unavailable')
        linked = r.headers.get('X-Linked-Etag', '').strip('"')
        if linked != self.files[name]['lfs']['sha256']:
            raise ValueError('Resolved file identity differs from pin')
        n = struct.unpack('<Q', self.bytes(name, 0, 8))[0]
        if not 0 < n <= MAX_BOUNDARY: raise ValueError('Unexpected header size')
        h = json.loads(self.bytes(name, 8, 8+n))
        result = dict(repo=self.repo, revision=self.rev, file=name, xet_hash=xet_hash,
                      lfs_sha256=linked, start=8+n, size=self.files[name]['size'],
                      tensors={k:v for k,v in h.items() if k != '__metadata__'})
        atomic_json(cache, result)
        return result

    def gap(self, info, key):
        spec = info['tensors'][key]
        s, e = (info['start'] + v for v in spec['data_offsets'])
        if e-s <= MAX_BOUNDARY:
            return dict(start=s, end=e, gap_start=s, gap_end=s, nodes=[])
        cache = self.cache / (hashlib.sha256(key.encode()).hexdigest() + '.gap.json')
        if cache.exists(): return json.loads(cache.read_text())
        cas, headers = self.auth()
        # Mark the complement dirty; the remaining gap is a server-certified
        # sequence of content hashes entirely inside the requested tensor.
        ranges = [f'0-{s-1}']
        if e < info['size']: ranges.append(f'{e}-{info["size"]-1}')
        headers['X-Range-Dirty'] = 'bytes=' + ','.join(ranges)
        r = self.get(cas + '/v2/file-chunk-hashes/' + info['xet_hash'], headers=headers).json()
        if r['fileSize'] != info['size'] or not r['windows'] or r['windows'][0]['dirtyByteRange'][0] != 0:
            raise ValueError('Unexpected CAS gap response')
        if len(r['windows']) > 2: raise ValueError('Unexpected gap count')
        gs = r['windows'][0]['dirtyByteRange'][1]
        ge = r['windows'][1]['dirtyByteRange'][0] if len(r['windows']) == 2 else info['size']
        nodes = (r['hashRanges'][1] or {}).get('nodes', [])
        if not s <= gs <= ge <= e or gs-s > MAX_BOUNDARY or e-ge > MAX_BOUNDARY:
            raise ValueError(f'CAS boundaries exceed limit: {self.side} {key} tensor={s}:{e} gap={gs}:{ge}')
        if sum(x['size'] for x in nodes) != ge-gs or any(len(x['hash']) != 64 or x['size'] <= 0 for x in nodes):
            raise ValueError('CAS node coverage mismatch')
        result = dict(start=s, end=e, gap_start=gs, gap_end=ge, nodes=nodes)
        atomic_json(cache, result)
        return result


def compare(root, original, donor, name):
    dest = root / 'audit' / 'xet-receipts' / (name + '.json')
    if dest.exists(): return json.loads(dest.read_text())
    a, b = original.file(name), donor.file(name)
    if a['tensors'].keys() != b['tensors'].keys(): raise ValueError('Tensor inventory differs')
    results = []
    for key, spec in a['tensors'].items():
        other = b['tensors'][key]
        if any(spec[k] != other[k] for k in ['dtype', 'shape']): raise ValueError('Tensor shape/dtype differs: '+key)
        size = spec['data_offsets'][1] - spec['data_offsets'][0]
        if size != other['data_offsets'][1] - other['data_offsets'][0]: raise ValueError('Tensor length differs')
        ga, gb = original.gap(a, key), donor.gap(b, key)
        same_layout = (ga['gap_start']-ga['start'], ga['gap_end']-ga['start']) == (gb['gap_start']-gb['start'], gb['gap_end']-gb['start'])
        equal_interior = same_layout and ga['nodes'] == gb['nodes']
        row = dict(name=key, dtype=spec['dtype'], shape=spec['shape'], bytes=size,
                   shard=name, original_offset=ga['start'], huihui_offset=gb['start'],
                   status='requires_tensor_bytes', reason='interior_hash_or_chunk_alignment_differs')
        if equal_interior:
            edge_hashes = {}
            edge_equal = True
            for label, begin_key, end_key in [('prefix', 'start', 'gap_start'), ('suffix', 'gap_end', 'end')]:
                aa = original.bytes(name, ga[begin_key], ga[end_key])
                bb = donor.bytes(name, gb[begin_key], gb[end_key])
                edge_equal = edge_equal and aa == bb
                edge_hashes[label] = dict(bytes=len(aa), original_sha256=hashlib.sha256(aa).hexdigest(), huihui_sha256=hashlib.sha256(bb).hexdigest())
            row.update(status='equal' if edge_equal else 'requires_tensor_bytes',
                       reason='xet_subtrees_and_boundary_bytes_equal' if edge_equal else 'boundary_bytes_differ',
                       boundary_hashes=edge_hashes, interior_bytes=ga['gap_end']-ga['gap_start'])
        results.append(row)
    result = dict(shard=name, original_xet_hash=a['xet_hash'], huihui_xet_hash=b['xet_hash'], tensors=results)
    dest.parent.mkdir(exist_ok=True, parents=True)
    atomic_json(dest, result)
    needed = [r for r in results if r['status'] != 'equal']
    print('XET AUDITED', name, 'equal', len(results)-len(needed), 'needs_bytes', len(needed), 'MiB', round(sum(r['bytes'] for r in needed)/2**20,2), flush=True)
    return result


def run(root, workers):
    global network_reserved
    network_reserved = sum(p.stat().st_size for p in (root/'audit'/'xet').rglob('*.range'))
    a, b = Remote(root, 'original'), Remote(root, 'huihui')
    idx = [json.loads((root/'metadata'/s/'model.safetensors.index.json').read_text())['weight_map'] for s in ['original','huihui']]
    if idx[0] != idx[1]: raise ValueError('Shard inventories differ')
    names = sorted(set(idx[0].values()))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda name: compare(root, a, b, name), names))
    rows = [r for shard in results for r in shard['tensors']]
    if len(rows) != len(idx[0]) or {r['name'] for r in rows} != set(idx[0]): raise ValueError('Incomplete audit')
    needed = [r for r in rows if r['status'] != 'equal']
    result = dict(status='metadata_audit_complete', method='Xet content subtrees plus exact boundary-byte comparison',
                  sources={s:manifest(root,s)[0]['sha'] for s in ['original','huihui']},
                  total_tensors=len(rows), equal_tensors=len(rows)-len(needed),
                  requires_bytes_tensors=len(needed), requires_bytes_both_sides=sum(r['bytes'] for r in needed)*2,
                  tensors=rows)
    atomic_json(root/'audit'/'xet-audit.json', result)
    print('XET AUDIT COMPLETE', len(rows), 'needs_bytes', len(needed), 'both_sides_GiB', result['requires_bytes_both_sides']/2**30, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.root,a.workers)
