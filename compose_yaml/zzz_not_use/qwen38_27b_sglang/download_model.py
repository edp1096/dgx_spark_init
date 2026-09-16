#!/usr/bin/env python3
"""Download pinned, ready-to-run weights. Reuse verified local weight shards."""
import hashlib
import json
import os
from pathlib import Path
import sys
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def download(repo, revision, dest, token):
    info = HfApi(token=token).model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError("Expected a pinned repository commit")
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".sparktalk-download.json"
    files = [f for f in info.siblings if not f.rfilename.startswith('.')]
    try:
        saved = json.loads(marker.read_text())
    except (OSError, ValueError):
        saved = {}
    stats = {f.rfilename: [p.stat().st_size, p.stat().st_mtime_ns] for f in files if (p := dest / f.rfilename).is_file()}
    if saved == {"repo": repo, "revision": revision, "files": stats} and len(stats) == len(files):
        print("Already prepared: " + repo, flush=True)
        return
    for i, f in enumerate(files, 1):
        path = dest / f.rfilename
        print(f"[{i}/{len(files)}] {f.rfilename}", flush=True)
        force = False
        if path.is_file() and f.lfs and path.stat().st_size == f.size:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest == f.lfs.sha256:
                print("Verified existing file", flush=True)
                continue
            force = True
        hf_hub_download(repo, f.rfilename, revision=revision, local_dir=dest, token=token, force_download=force)
    index = json.loads((dest / "model.safetensors.index.json").read_text())
    for filename in set(index['weight_map'].values()):
        if not (dest / filename).is_file():
            raise RuntimeError("Missing weight shard: " + filename)
    stats = {f.rfilename: [(dest / f.rfilename).stat().st_size, (dest / f.rfilename).stat().st_mtime_ns] for f in files}
    temp = marker.with_suffix('.tmp')
    temp.write_text(json.dumps({"repo": repo, "revision": revision, "files": stats}))
    temp.replace(marker)


def main():
    token = sys.stdin.readline().strip() or None
    download(os.environ['MODEL_REPO'], os.environ['MODEL_REVISION'], Path('/model'), token)
    print('Preparing DFlash2', flush=True)
    snapshot_download(os.environ['DRAFT_REPO'], revision=os.environ['DRAFT_REVISION'], token=token)
    print('Model and DFlash2 ready; no quantization or ablation patch required.', flush=True)

if __name__ == '__main__':
    main()
