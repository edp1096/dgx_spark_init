#!/usr/bin/env python3
"""Standalone preparation, conversion and verification for the allocation-reuse experiment."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request
from preflight import inspect_model

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('DATA_ROOT', ROOT / 'data'))
CFG = json.loads((ROOT / 'settings.json').read_text())
LLAMA = Path('/opt/llama.cpp')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    partial.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def import_source(source):
    if (DATA / 'source/ready.json').exists():
        inspect_model(DATA / 'source')
        print('Local source is already prepared.', flush=True)
        return
    inventory = inspect_model(source)
    DATA.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(DATA).free < inventory['weight_bytes'] + (130 << 30):
        raise RuntimeError('Need source size plus 130 GiB free for conversion and output')
    staging = DATA / 'source-import'
    staging.mkdir(exist_ok=True)
    url = f'https://huggingface.co/api/models/{CFG["source_repo"]}/tree/{CFG["source_revision"]}?recursive=true&limit=1000'
    with urllib.request.urlopen(url, timeout=30) as response:
        upstream = json.load(response)
    expected_weights = {item['path']: item['lfs']['oid'] for item in upstream if item['path'].endswith('.safetensors')}
    files = [p for p in source.iterdir() if p.is_file() and (
        p.suffix in {'.json', '.safetensors', '.txt', '.jinja', '.model'} or p.name in {'LICENSE', 'README.md'})]
    hashes = {}
    for file in sorted(files):
        target = staging / file.name
        if not target.exists() or target.stat().st_size != file.stat().st_size:
            print('Copying', file.name, flush=True)
            partial = target.with_name(target.name + '.partial')
            shutil.copyfile(file, partial)
            partial.replace(target)
        hashes[file.name] = digest(target)
        if file.suffix == '.safetensors' and hashes[file.name] != expected_weights.get(file.name):
            raise ValueError(f'{file.name}: SHA256 does not match the pinned Huihui revision')
    result = inspect_model(staging)
    save(staging / 'ready.json', {'repo': CFG['source_repo'], 'revision': CFG['source_revision'],
                                'source': inventory, 'copied_files_sha256': hashes,
                                'upstream_weight_sha256_verified': expected_weights})
    destination = DATA / 'source'
    if destination.exists():
        raise RuntimeError('Incomplete data/source exists; inspect it before importing')
    staging.rename(destination)
    print('Prepared source:', result['tensor_count'], 'BF16 tensors', flush=True)


def prepare_source():
    if (DATA / 'source/ready.json').exists():
        return inspect_model(DATA / 'source')
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(CFG['source_repo'], revision=CFG['source_revision'],
                                 cache_dir=DATA / 'download-cache',
                                 allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model', 'LICENSE', 'README.md'])
    import_source(Path(snapshot))
    return inspect_model(DATA / 'source')


def allocation():
    result = {}
    for line in (DATA / 'reference/allocation.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, qtype = [part.strip() for part in line.split(':', 1)]
        if name in result or not re.fullmatch(r'[A-Za-z0-9_.]+', name) or not re.fullmatch(r'[A-Z0-9_]+', qtype):
            raise ValueError('Invalid or duplicate allocation entry')
        result[name] = qtype
    if len(result) != 866 or len([name for name in result if name.startswith('blk.64.')]) != 15:
        raise ValueError('Expected 866 tensors including the 15-tensor MTP block')
    return result


def prepare_reference():
    folder = DATA / 'reference'
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for remote, local in [(CFG['reference_allocation'], 'allocation.txt'), (CFG['reference_imatrix'], 'imatrix.gguf')]:
        target = folder / local
        if not target.exists():
            url = f'https://huggingface.co/{CFG["reference_repo"]}/resolve/{CFG["reference_revision"]}/{remote}'
            print('Downloading reference', remote, flush=True)
            with urllib.request.urlopen(url, timeout=120) as stream, target.with_suffix('.partial').open('wb') as dest:
                shutil.copyfileobj(stream, dest)
            target.with_suffix('.partial').replace(target)
        hashes[local] = digest(target)
        if hashes[local] != CFG['reference_sha256'][local]:
            raise ValueError(f'{local}: reference checksum mismatch')
    types = allocation()
    # llama-quantize uses regex_search: anchor and escape every exact tensor name.
    (folder / 'tensor-types.txt').write_text(''.join(f'^{re.escape(name)}$={qtype}\n' for name, qtype in types.items()))
    save(folder / 'manifest.json', {'repo': CFG['reference_repo'], 'revision': CFG['reference_revision'],
                                   'files_sha256': hashes, 'tensor_types': dict(collections.Counter(types.values())),
                                   'method': CFG['method']})


def run(name, command):
    logs = DATA / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['TMPDIR'] = str(DATA / 'tmp')
    env['HF_HUB_OFFLINE'] = '1'
    env['OMP_NUM_THREADS'] = '4'
    Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    start = time.time()
    print('Running', name, '; log:', logs / (name + '.log'), flush=True)
    with (logs / (name + '.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    save(logs / (name + '.json'), {'command': command, 'seconds': time.time() - start, 'exit_code': result.returncode})
    if result.returncode:
        raise RuntimeError(f'{name} failed; inspect {logs / (name + ".log")}')


def read_gguf(path, expected_types=None):
    import gguf
    reader = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError('Duplicate GGUF tensor names')
    if expected_types is not None:
        if set(tensors) != set(expected_types):
            raise ValueError(f'Tensor map mismatch; extra={set(tensors)-set(expected_types)}, missing={set(expected_types)-set(tensors)}')
        for name, target_type in expected_types.items():
            actual = tensors[name].tensor_type.name
            if actual != target_type:
                raise ValueError(f'{name}: expected {target_type}, got {actual}')
    def field(name):
        value = reader.fields.get(name)
        return value.contents() if value else None
    return {'bytes': path.stat().st_size, 'tensor_count': len(tensors),
            'architecture': field('general.architecture'),
            'model_name': field('general.name'),
            'block_count': field('qwen35.block_count'),
            'mtp_layers': field('qwen35.nextn_predict_layers'),
            'mtp_tensor_count': sum(name.startswith('blk.64.') for name in tensors),
            'types': dict(collections.Counter(t.tensor_type.name for t in tensors.values()))}


def convert():
    revision = (LLAMA / '.git/HEAD').read_text().strip()
    if revision != CFG['llama_commit']:
        raise RuntimeError('Converter image does not match the pinned llama.cpp revision')
    prepare_source()
    prepare_reference()
    work = DATA / 'work'
    models = DATA / 'models'
    work.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    for name, destination, extra in [
        ('convert-bf16', work / 'source-bf16-mtp.gguf', []),
        ('convert-mmproj', models / 'mmproj-Huihui-Qwen3.8-27B-BF16.gguf', ['--mmproj'])]:
        if not destination.exists():
            temporary = destination.with_name(destination.stem + '.partial.gguf')
            run(name, ['python3', str(ROOT / 'converter.py'), str(DATA / 'source'),
                       '--outtype', 'bf16', '--use-temp-file', '--model-name', CFG['model_name'],
                       '--outfile', str(temporary), *extra])
            read_gguf(temporary)
            temporary.replace(destination)
    # Validate every expected name before the expensive quantization step.
    import gguf
    reader = gguf.GGUFReader(work / 'source-bf16-mtp.gguf')
    actual = {t.name for t in reader.tensors}
    expected = set(allocation())
    if actual != expected:
        raise ValueError(f'Converted names differ from reference allocation: extra={actual-expected}, missing={expected-actual}')
    print('Converted BF16 and vision projector; tensor names match allocation.', flush=True)


def quantize():
    convert()
    output = DATA / 'models' / CFG['output_file']
    if not output.exists():
        temporary = output.with_name(output.stem + '.partial.gguf')
        run('quantize', ['llama-quantize', '--imatrix', str(DATA / 'reference/imatrix.gguf'),
                         '--tensor-type-file', str(DATA / 'reference/tensor-types.txt'),
                         '--max-buffer-size', '512', str(DATA / 'work/source-bf16-mtp.gguf'),
                         str(temporary), 'IQ3_S', os.environ.get('QUANT_THREADS', '12')])
        read_gguf(temporary, allocation())
        temporary.replace(output)
    verify()


def verify():
    output = DATA / 'models' / CFG['output_file']
    report = read_gguf(output, allocation())
    if report['mtp_tensor_count'] != 15 or report['architecture'] != 'qwen35':
        raise ValueError('Unexpected architecture or missing MTP')
    report['sha256'] = digest(output)
    projector = DATA / 'models/mmproj-Huihui-Qwen3.8-27B-BF16.gguf'
    report['projector'] = {**read_gguf(projector), 'sha256': digest(projector)}
    report['provenance'] = CFG
    report['recipe_sha256'] = {name: digest(ROOT / name) for name in ['Dockerfile', 'settings.json', 'pipeline.py', 'converter.py']}
    report['runtime_validation'] = 'not performed by this structural check'
    save(DATA / 'reports/model.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['import-source', 'prepare', 'convert', 'quantize', 'verify'])
    parser.add_argument('source', nargs='?', type=Path)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.stage == 'import-source':
        if args.source is None:
            parser.error('import-source requires a directory')
        import_source(args.source)
    elif args.stage == 'prepare':
        prepare_source()
        prepare_reference()
    else:
        {'convert': convert, 'quantize': quantize, 'verify': verify}[args.stage]()


if __name__ == '__main__':
    main()
