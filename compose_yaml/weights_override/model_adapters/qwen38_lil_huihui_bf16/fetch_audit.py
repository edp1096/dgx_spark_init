"""Pinned QAD download and shared receipt utilities. BF16 uses selective ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

CHUNK = 4 << 20


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def manifest(root, side):
    d = json.loads((root / 'metadata' / (side + '-api.json')).read_text())
    return d, {f['rfilename']: f for f in d['siblings']}


def fetch(d, record, dest):
    expected = record.get('lfs', {}).get('sha256')
    length = record['size']
    if dest.exists():
        if dest.stat().st_size != length or (expected and sha(dest) != expected):
            raise ValueError('Existing file differs from pinned input: ' + str(dest))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + '.partial')
    for attempt in range(6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > length:
                raise ValueError('Partial exceeds expected file size')
            if offset < length:
                url = f"https://huggingface.co/{d['id']}/resolve/{d['sha']}/{record['rfilename']}?download=true&t={time.time_ns()}"
                headers = {'Range': f'bytes={offset}-{length-1}'} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    if offset and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                        raise ValueError('Resume range was not honored')
                    with partial.open('ab' if offset else 'wb') as f:
                        for block in response.iter_content(CHUNK):
                            f.write(block)
            if partial.stat().st_size != length:
                raise ValueError('Downloaded size mismatch')
            if expected and sha(partial) != expected:
                partial.unlink()
                raise ValueError('Downloaded SHA256 mismatch')
            partial.replace(dest)
            print('FETCHED', record['rfilename'], length, flush=True)
            return
        except Exception as exc:
            print('RETRY', record['rfilename'], attempt, type(exc).__name__, str(exc)[:160], flush=True)
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))


def run(root, mode, workers):
    if mode != 'qad':
        raise ValueError('BF16 requires xet_audit.py and fetch_selected.py')
    d, files = manifest(root, 'qad')
    allowed = [r for n, r in files.items() if n.endswith(('.safetensors', '.json', '.jinja', '.txt', '.md')) or n == 'LICENSE']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda r: fetch(d, r, root / 'base' / r['rfilename']), allowed))
    atomic_json(root / 'base' / 'download-receipt.json', {'repo': d['id'], 'revision': d['sha'], 'status': 'verified', 'files': allowed})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--mode', choices=['qad'], required=True)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    run(a.root, a.mode, a.workers)
