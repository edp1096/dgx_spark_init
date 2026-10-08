"""Release only closed, unmapped weight-file cache after complete GPU load."""
import json
import os
from pathlib import Path
import urllib.request


def release_cache(model_directory, process=Path('/proc/1')):
    files = sorted(model_directory.glob('model-?????-of-00007.safetensors'))
    if len(files) != 7 or any(not path.is_file() for path in files):
        raise RuntimeError('Expected exactly seven original non-PLE checkpoint files')
    resolved = {str(path.resolve()) for path in files}
    mappings = (process / 'maps').read_text()
    descriptors = {str(path.resolve()) for path in (process / 'fd').iterdir()}
    if resolved.intersection(descriptors) or any(path in mappings for path in resolved):
        raise RuntimeError('Checkpoint files are still open or mapped; cache was not released')
    before = Path('/proc/meminfo').read_text()
    for path in files:
        with path.open('rb') as source:
            os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return {'files': [path.name for path in files], 'policy': 'closed non-PLE file cache only',
            'before': before, 'after': Path('/proc/meminfo').read_text()}


if __name__ == '__main__':
    with urllib.request.urlopen('http://127.0.0.1:30000/v1/model', timeout=10) as response:
        card = json.load(response)
    parameters = card['parameters']
    if card['id'] != 'qwen38fn_exl3' or parameters['max_seq_len'] != 1048576 or parameters['cache_size'] != 1048576 or parameters['cache_mode'] != 'Q8' or not parameters['use_vision']:
        raise RuntimeError('Loaded qwen38fn_exl3 profile does not match the qualified 1M runtime')
    result = release_cache(Path('/runtime/models/qwen38fn_exl3'))
    Path('/runtime/weight-cache-release.json').write_text(json.dumps(result, indent=2))
    print('SPARKTALK_QWEN38FN_EXL3_WEIGHT_CACHE_RELEASED files=7', flush=True)
