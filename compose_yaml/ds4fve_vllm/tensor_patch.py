"""Reconstruct pinned HF blobs from local originals and small verified patches."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def assemble(original, target, header, patches, expected):
    target = Path(target)
    if target.exists() and digest(target) == expected: return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.patching')
    try:
        with open(original, 'rb') as src, open(temporary, 'wb') as dst:
            old_length = struct.unpack('<Q', src.read(8))[0]
            old = json.loads(src.read(old_length))
            new = json.loads(header[8:])
            old.pop('__metadata__', None); new.pop('__metadata__', None)
            if old != new: raise ValueError('Tensor layout changed; selective patch refused')
            dst.write(header)
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            for offset, patch in patches:
                dst.seek(len(header) + offset)
                with open(patch, 'rb') as p: shutil.copyfileobj(p, dst)
        if digest(temporary) != expected:
            raise ValueError('Patched shard SHA-256 mismatch; refusing full-shard fallback')
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def apply_bundle(bundle, home):
    bundle, home = Path(bundle), Path(home)
    manifest = json.loads((bundle/'manifest.json').read_text())
    root = home/'hub'/('models--'+manifest['repo'].replace('/', '--'))
    original = home/'hub'/('models--'+manifest['original'].replace('/', '--'))/'blobs'
    snapshot = root/'snapshots'/manifest['revision']
    for i, entry in enumerate(manifest['files'], 1):
        target = root/'blobs'/entry['hash']; target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[로컬 구성 {i}/{len(manifest['files'])}] {entry['name']}", flush=True)
        if 'header' in entry:
            assemble(original/entry['source'], target, (bundle/entry['header']).read_bytes(),
                     [(p['offset'], bundle/p['file']) for p in entry['patches']], entry['hash'])
        elif 'source' in entry:
            source = original/entry['source']
            if not source.exists(): raise ValueError('Original cache missing: '+entry['name'])
            if not target.exists(): os.link(source, target)
        elif not target.exists():
            shutil.copyfile(bundle/entry['file'], target)
        link = snapshot/entry['name']; link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists(): link.symlink_to(os.path.relpath(target, link.parent))
    (root/'refs').mkdir(exist_ok=True)
    (root/'refs'/'main').write_text(manifest['revision']+'\n')
    print('선택 텐서 모델 구성 완료', flush=True)


if __name__ == '__main__': apply_bundle(sys.argv[1], sys.argv[2])
