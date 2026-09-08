"""Download selected revision; reuse only hash-identical official blobs for ablit."""
import os
import json
import struct
import requests
import shutil
from tensor_patch import apply_bundle
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

home = Path(os.environ['HF_HOME'])
token_file = Path('/run/secrets/hf_token')
token = token_file.read_text().strip() if token_file.is_file() else None
variant = os.environ['MODEL_VARIANT']
repo = os.environ['DSPARK_MODEL_ABLITERATED' if variant == 'abliterated' else 'DSPARK_MODEL_OFFICIAL']
revision = os.environ['DSPARK_REVISION_ABLITERATED' if variant == 'abliterated' else 'DSPARK_REVISION']
if not revision: raise SystemExit('A pinned model revision is required')
info = HfApi(token=token).model_info(repo, revision=revision, files_metadata=True)
root = home/'hub'/('models--'+repo.replace('/', '--'))
original = home/'hub'/('models--'+os.environ['DSPARK_MODEL_OFFICIAL'].replace('/', '--'))/'blobs'
snapshot = root/'snapshots'/info.sha
if variant == 'abliterated':
    # Pinned recipe: validate tensor names and every reconstructed shard against HF SHA-256.
    original_info = HfApi(token=token).model_info(os.environ['DSPARK_MODEL_OFFICIAL'], revision=os.environ['DSPARK_REVISION'], files_metadata=True)
    originals = {e.rfilename: e for e in original_info.siblings}
    bundle = root/'tensor-patches'/info.sha
    bundle.mkdir(parents=True, exist_ok=True)
    def fetch_range(name, start, end):
        url = f'https://huggingface.co/{repo}/resolve/{info.sha}/{name}?range={start}-{end}'
        with requests.get(url, headers={'Authorization': 'Bearer '+token, 'Range': f'bytes={start}-{end}'}, stream=True, timeout=120) as response:
            response.raise_for_status()
            if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                raise ValueError('Server ignored range; refusing full-shard download')
            data = response.raw.read(end-start+2)
            if len(data) != end-start+1: raise ValueError('Truncated or oversized range')
            return data
    manifest = {'repo':repo, 'original':os.environ['DSPARK_MODEL_OFFICIAL'], 'revision':info.sha, 'files':[]}
    for index, entry in enumerate(info.siblings, 1):
        name = entry.rfilename
        if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('Unsafe path')
        blob = entry.lfs.sha256 if entry.lfs else entry.blob_id
        source_entry = originals.get(name)
        source_hash = (source_entry.lfs.sha256 if source_entry.lfs else source_entry.blob_id) if source_entry else None
        record = {'name':name, 'hash':blob}
        print(f'[선택 준비 {index}/{len(info.siblings)}] {name}', flush=True)
        if source_hash == blob and (original/source_hash).is_file():
            record['source'] = source_hash
        elif name.endswith('.safetensors'):
            if not source_hash or not (original/source_hash).is_file():
                raise ValueError('Original shard required for selective patch: '+name)
            first = fetch_range(name, 0, 7)
            length = struct.unpack('<Q', first)[0]
            if length > 16*1024*1024: raise ValueError('Oversized tensor header')
            header = first + fetch_range(name, 8, 7+length)
            header_name = blob+'.header'
            (bundle/header_name).write_bytes(header)
            layout = json.loads(header[8:])
            edits = []
            for layer in range(10,36):
                for suffix in ['weight', 'scale']:
                    key = f'layers.{layer}.attn.wo_b.{suffix}'
                    if key not in layout: continue
                    start,end = layout[key]['data_offsets']
                    patch_name = blob+'.'+key+'.bin'
                    patch = bundle/patch_name
                    if not patch.exists() or patch.stat().st_size != end-start:
                        print(f'텐서 다운로드: {key} ({(end-start)/2**20:.2f} MiB)', flush=True)
                        data = fetch_range(name, len(header)+start, len(header)+end-1)
                        temporary = patch.with_suffix('.tmp'); temporary.write_bytes(data); temporary.replace(patch)
                    edits.append({'offset':start,'file':patch_name})
            if not edits: raise ValueError('Unrecognized changed shard: '+name)
            record.update(source=source_hash, header=header_name, patches=edits)
        else:
            downloaded = hf_hub_download(repo, name, revision=info.sha, token=token)
            filename = blob+'.file'; shutil.copyfile(downloaded, bundle/filename); record['file']=filename
        manifest['files'].append(record)
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    apply_bundle(bundle, home)
    raise SystemExit(0)

total = len(info.siblings)
for index, entry in enumerate(info.siblings, 1):
    print(f"[파일 {index}/{total}] 준비 시작: {entry.rfilename}", flush=True)
    rel = Path(entry.rfilename)
    if rel.is_absolute() or '..' in rel.parts: raise SystemExit('Unsafe Hub file path')
    blob_id = entry.lfs.sha256 if entry.lfs else entry.blob_id
    source = original/blob_id if blob_id else None
    if variant == 'abliterated' and source and source.is_file() and source.stat().st_size == entry.size:
        blob = root/'blobs'/blob_id
        blob.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists(): os.link(source, blob)
        link = snapshot/rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists(): link.symlink_to(os.path.relpath(blob, link.parent))
        print('shared:', rel, flush=True)
    else:
        print('download:', rel, flush=True)
        hf_hub_download(repo, str(rel), revision=info.sha, token=token)
    print(f'[파일 완료 {index}/{total}] {index * 100 // total}% (파일 개수 기준)', flush=True)
refs = root/'refs'
refs.mkdir(parents=True, exist_ok=True)
(refs/'main').write_text(info.sha+'\n')
print('Model prepared:', repo, info.sha, flush=True)
