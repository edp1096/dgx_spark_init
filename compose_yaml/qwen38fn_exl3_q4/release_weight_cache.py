"""Verify the live profile and return only closed, unmapped non-PLE weight pages."""
import json
import os
from pathlib import Path
import urllib.request
from launch import MODEL, MODEL_ID, command


def release_cache():
    with urllib.request.urlopen('http://127.0.0.1:30000/v1/models', timeout=10) as response:
        models = json.load(response)['data']
    if len(models) != 1 or models[0]['id'] != MODEL_ID or models[0]['max_model_len'] != 1048576:
        raise RuntimeError('Actual Velo model or context does not match the qualified 1M profile')
    args = Path('/proc/1/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if args != command():
        raise RuntimeError('Actual Velo launch differs from the qualified GPU/Q8 KV/YaRN/MTP profile')
    files = sorted(MODEL.glob('model-?????-of-00009.safetensors'))
    resolved = {str(path.resolve()) for path in files}
    for process in Path('/proc').glob('[0-9]*'):
        try:
            mappings = (process / 'maps').read_text()
            descriptors = {str(p.resolve()) for p in (process / 'fd').iterdir()}
        except FileNotFoundError:
            continue
        if resolved.intersection(descriptors) or any(path in mappings for path in resolved):
            raise RuntimeError('A main checkpoint shard is still mapped/open; refusing cache release')
    for path in files:
        with path.open('rb') as source:
            os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    Path('/runtime/weight-cache-release.json').write_text(json.dumps({'files': [p.name for p in files], 'profile': '1M Q8 KV GPU vision MTP3 full vocabulary'}))
    print('SPARKTALK_EXL3_Q4_READY files=9', flush=True)


if __name__ == '__main__':
    release_cache()
