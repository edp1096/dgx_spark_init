"""CPU-only, resumable preparation. Credentials arrive on stdin, never argv."""
import hashlib
import json
import os
import struct
import sys
from pathlib import Path


def complete(root, item):
    try:
        if item.get('pipeline'):
            marker = root / '.sparktalk-complete.json'
            if not marker.is_file() or not (root / 'model_index.json').is_file():
                return False
            entries = json.loads(marker.read_text())
            return len(entries)>1 and all(not Path(n).is_absolute() and '..' not in Path(n).parts and (root/n).is_file() and (root/n).stat().st_size==size for n,size in entries.items())
        if item.get('files'):
            for name in item['files']:
                p = root / name
                if not p.is_file() or p.stat().st_size == 0:
                    return False
                if name in item.get('sha256', {}):
                    with p.open('rb') as f:
                        if hashlib.file_digest(f, 'sha256').hexdigest() != item['sha256'][name]:
                            return False
                if name.endswith('.gguf'):
                    with p.open('rb') as f:
                        if f.read(4) != b'GGUF':
                            return False
            return True
        if not (root / 'config.json').is_file():
            return False
        json.loads((root / 'config.json').read_text())
        indices = list(root.glob('*.safetensors.index.json'))
        if indices:
            shards = set()
            for index in indices:
                shards.update(json.loads(index.read_text())['weight_map'].values())
        else:
            shards = {p.name for p in root.glob('*.safetensors')}
        if not shards:
            return False
        for name in shards:
            if Path(name).name != name:
                return False
            with (root / name).open('rb') as f:
                raw = f.read(8)
                if len(raw) != 8:
                    return False
                length = struct.unpack('<Q', raw)[0]
                if length > 64 * 1024 * 1024:
                    return False
                header = json.loads(f.read(length))
                end = max((v['data_offsets'][1] for k, v in header.items() if k != '__metadata__'), default=0)
                if os.fstat(f.fileno()).st_size != 8 + length + end:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare(item, token):
    from huggingface_hub import hf_hub_download, snapshot_download
    root = Path(item['path'])
    if complete(root, item):
        print('Already prepared: ' + str(root), flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    options = dict(repo_id=item['repo'], revision=item.get('revision', 'main'), token=token or False)
    print('Preparing: ' + item['repo'], flush=True)
    if item.get('files'):
        for name in item['files']:
            if complete(root, {**item, 'files': [name]}):
                continue
            hf_hub_download(filename=name, local_dir=root, force_download=(root/name).exists(), **options)
    elif item.get('hub_cache'):
        snapshot_download(cache_dir='/hf/hub', max_workers=1, **options)
    else:
        snapshot_download(local_dir=root, max_workers=1, **options)
    if item.get('pipeline'):
        entries = {str(p.relative_to(root)):p.stat().st_size for p in root.rglob('*') if p.is_file() and p.name!='.sparktalk-complete.json' and '.cache' not in p.relative_to(root).parts}
        (root/'.sparktalk-complete.json').write_text(json.dumps(entries))
    if not complete(root, item):
        raise RuntimeError('Incomplete checkpoint: ' + str(root))
    print('Ready: ' + str(root), flush=True)


def main():
    payload = json.load(sys.stdin)
    if payload.get('check_only'):
        sys.exit(0 if all(complete(Path(i['path']), i) for i in payload['items']) else 1)
    for item in payload['items']:
        prepare(item, payload.get('token', ''))


if __name__ == '__main__':
    main()
