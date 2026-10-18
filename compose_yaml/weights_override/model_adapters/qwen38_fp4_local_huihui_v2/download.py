"""Download a pinned FP4-Local checkpoint with bounded, resumable writes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import threading

import requests


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    with path.open('rb') as f:
        result = hashlib.file_digest(f, 'sha256').hexdigest()
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return result


def fetch(session, api, record, destination, progress):
    size = record['size']
    expected = record.get('lfs', {}).get('sha256')
    if destination.exists():
        actual = digest(destination)
        if destination.stat().st_size != size or expected and actual != expected:
            raise ValueError('Existing pinned file failed verification: ' + record['rfilename'])
        return actual
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.partial')
    last_progress = 0
    for attempt in range(6):
        done = partial.stat().st_size if partial.exists() else 0
        if done > size:
            raise ValueError('Partial exceeds pinned size')
        try:
            if done < size:
                url = f"https://huggingface.co/{api['id']}/resolve/{api['sha']}/{record['rfilename']}?download=true&resume={done}"
                headers = {'Range': f'bytes={done}-{size-1}'} if done else {}
                with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if done and (response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {done}-{size-1}/{size}'):
                        raise ValueError('Resume range not honored')
                    if not done and response.status_code != 200:
                        raise ValueError('Unexpected initial download status')
                    with partial.open('ab' if done else 'wb') as out:
                        for block in response.iter_content(4 << 20):
                            if done + len(block) > size:
                                raise ValueError('Download exceeds pinned size')
                            out.write(block)
                            done += len(block)
                            if time.monotonic() - last_progress > 10:
                                progress(record['rfilename'], done, size)
                                last_progress = time.monotonic()
            if done != size:
                raise ValueError('Truncated download')
            actual = digest(partial)
            if expected and actual != expected:
                raise ValueError('Pinned SHA256 mismatch: ' + record['rfilename'])
            partial.replace(destination)
            progress(record['rfilename'], size, size)
            return actual
        except requests.RequestException as exc:
            # Signed URLs/tokens from transport exceptions must not enter logs.
            print('RETRY', record['rfilename'], attempt + 1, type(exc).__name__, flush=True)
            if attempt == 5:
                raise RuntimeError('Download transport failed: ' + record['rfilename']) from None
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError('Download did not finish')


def run(root, workers=2):
    api = json.loads((root / 'metadata/base-api.json').read_text())
    if api['sha'] != 'b9ce18a6887e18c089fdea8509a33e63429a3300':
        raise ValueError('Unexpected FP4-Local revision')
    files = [f for f in api['siblings'] if f['rfilename'] != '.gitattributes']
    # Metadata first; tensors then remain in upstream order.
    priority = {'tensors/model-00001-of-00021.safetensors': 0,
                'tensors/model-00015-of-00021.safetensors': 1}
    files.sort(key=lambda r: (r['rfilename'].endswith('.safetensors'),
                             priority.get(r['rfilename'], 2), r['rfilename']))
    started = time.monotonic()
    complete = []
    active = {}
    lock = threading.Lock()
    total = sum(f['size'] for f in files)

    def progress(name, done, size):
        with lock:
            active[name] = {'bytes': done, 'total': size}
            atomic(root / 'download-progress.json', {
                'active': dict(active),
                'completed_files': len(complete), 'total_files': len(files),
                'bytes': sum(f['size'] for f in complete) + sum(v['bytes'] for v in active.values()),
                'total_bytes': total, 'seconds': time.monotonic() - started,
            })

    def one(record):
        with requests.Session() as session:
            actual = fetch(session, api, record, root / 'base' / record['rfilename'], progress)
        with lock:
            complete.append({**record, 'verified_sha256': actual})
            active.pop(record['rfilename'], None)
        print('VERIFIED', record['rfilename'], record['size'], flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, f) for f in files]):
            future.result()
    atomic(root / 'base/download-receipt.json', {
        'status': 'verified', 'repo': api['id'], 'revision': api['sha'],
        'files': sorted(complete, key=lambda f: f['rfilename']),
        'seconds': time.monotonic() - started,
    })
    print('CHECKPOINT VERIFIED', total, 'bytes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    run(args.root, args.workers)
