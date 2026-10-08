"""Standalone Compose preparation; no SparkTalk files or services required."""
import hashlib
import json
import os
import struct
import sys
from pathlib import Path


def complete(root, identity=None):
    try:
        for name, expected in (identity or {}).items():
            if Path(name).name != name:
                return False
            with (root / name).open('rb') as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
                    return False
        json.loads((root / 'config.json').read_text())
        indices = list(root.glob('*.safetensors.index.json'))
        shards = set()
        for index in indices:
            shards.update(json.loads(index.read_text())['weight_map'].values())
        if not indices:
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


def prepare(repo, revision, root, hub, check_only=False):
    release = json.loads(Path(__file__).with_name('checkpoint.huihui-lil.json').read_text())
    identity = {}
    if repo == 'edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4':
        release = json.loads(Path(__file__).with_name('checkpoint.radixark.json').read_text())
        check_only = True
    if repo == release['repo']:
        if revision != release['revision']:
            raise ValueError('Checkpoint revision must match its pinned manifest')
        identity = release['sha256']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('A pinned 40-character model revision is required')
    if complete(root, identity):
        print('Ready: ' + repo + '@' + revision, flush=True)
        return True
    if check_only:
        print('Missing, incomplete, or different checkpoint: ' + str(root), flush=True)
        return False
    from huggingface_hub import snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    options = dict(repo_id=repo, revision=revision, max_workers=1)
    expected_snapshot = hub / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
    if root == expected_snapshot:
        snapshot_download(cache_dir=hub, **options)
    else:
        snapshot_download(local_dir=root, **options)
    if not complete(root, identity):
        raise RuntimeError('Incomplete or wrong checkpoint: ' + str(root))
    print('Ready: ' + repo + '@' + revision, flush=True)
    return True


def main():
    # Compose resolves all three from the same selected model env file.
    ready = prepare(os.environ['QWEN_QAD_MODEL_ID'], os.environ['QWEN_QAD_REVISION'],
                    Path(os.environ['QWEN_QAD_MODEL_PATH']), Path(os.environ.get('HF_HOME', '/hf')) / 'hub',
                    check_only='--check' in sys.argv[1:])
    return 0 if ready else 1


if __name__ == '__main__':
    sys.exit(main())
