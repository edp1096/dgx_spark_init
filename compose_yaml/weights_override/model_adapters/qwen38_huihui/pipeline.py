"""Single-entry orchestration for the existing Qwen GGUF-delta converter."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def input_paths(args, profile):
    paths = {}
    for key, (repo, revision) in profile['sources'].items():
        value = getattr(args, key, None)
        paths[key] = (value or args.cache / 'hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision).resolve()
    for key in ('original', 'donor'):
        directory = paths[key]
        for i in range(1, 5):
            file = directory / f'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-{i:05d}-of-00004.gguf'
            if not file.is_file():
                raise ValueError(f'Missing {key} GGUF: {file}')
        vision = directory / ('mmproj-BF16.gguf' if key == 'original' else 'mmproj-model-bf16.gguf')
        if not vision.is_file():
            raise ValueError(f'Missing vision GGUF: {vision}')
    base = paths['base']
    from weights_core.formats import validate_format
    validate_format(base, args.format)
    config = json.loads((base / 'config.json').read_text())
    if config.get('quantization_config', {}).get('quant_algo') != 'NVFP4':
        raise ValueError('Qwen GGUF profile requires the RadixArk NVFP4 layout')
    if config.get('architectures') != [profile['architecture']] or any(config.get('text_config', {}).get(k) != v for k,v in profile['text_config'].items()):
        raise ValueError('Base model does not match the Qwen3.8 Flash-Next profile')
    index = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    for name in set(index.values()):
        if Path(name).name != name or not (base / name).is_file():
            raise ValueError(f'Missing or invalid base shard: {name}')
    if args.source_raw is not None and args.source_bf16 is not None:
        raise ValueError('Choose --source-bf16 or --source-raw, not both')
    for key in ('source_raw', 'source_bf16'):
        value = getattr(args, key)
        if value is not None:
            paths[key] = value.resolve()
            if not paths[key].is_dir():
                raise ValueError(f'Missing BF16 expert source: {paths[key]}')
    output = args.output.resolve()
    for directory in paths.values():
        if output == directory or directory in output.parents or output in directory.parents:
            raise ValueError('Output must be separate from all input directories')
    if output.exists() or args.output.is_symlink():
        raise ValueError('Output already exists')
    if not args.check_only and args.activation_scales != 'preserve':
        raise ValueError('Qwen conversion requires --activation-scales preserve (no new calibration)')
    return paths


def run(args, profile):
    paths = input_paths(args, profile)
    image = args.helper_image or profile['helper_image']
    subprocess.run(['docker', 'image', 'inspect', image], check=True, stdout=subprocess.DEVNULL)
    gguf = args.gguf_package.resolve()
    if not (gguf / '__init__.py').is_file():
        raise ValueError(f'Local gguf package is missing: {gguf}; set --gguf-package')
    output = args.output.resolve()
    if args.check_only:
        job = Path(tempfile.mkdtemp(prefix='qwen-check-'))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        job = output.with_name('.' + output.name + '.conversion')
        job.mkdir()  # Exclusive: never reuse another run or a failed conversion.
    request = {k: str(v) for k, v in paths.items()}
    request.update(check_only=args.check_only, cache=str(args.cache.resolve()))
    (job / 'request.json').write_text(json.dumps(request))
    command = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
               '--read-only', '--memory', '6g', '--cpus', '2', '--user', f'{os.getuid()}:{os.getgid()}',
               '--tmpfs', '/tmp:rw,size=1g', '--entrypoint', 'python3',
               '-e', 'PYTHONPATH=/tool:/opt:/tool/vendor/Model-Optimizer',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=2', '-e', 'OPENBLAS_NUM_THREADS=2']
    mounts = [(ROOT, '/tool', True), (gguf, '/opt/gguf', True), (job, '/job', False)]
    # HF snapshots contain symlinks to blobs: mount their common cache root too.
    if args.cache.is_dir():
        mounts.append((args.cache.resolve(), str(args.cache.resolve()), True))
    mounts.extend((directory, str(directory), True) for directory in paths.values())
    for source, target, readonly in mounts:
        if ',' in str(source) or ',' in target:
            raise ValueError('Docker bind paths cannot contain commas')
        command += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if readonly else '')]
    command += [image, '/tool/model_adapters/qwen38_huihui/pipeline.py', '/job/request.json']
    try:
        subprocess.run(command, check=True)
        result = json.loads((job / 'result.json').read_text())
        if result['status'] != 'passed':
            raise ValueError('Qwen adapter verification did not pass')
        if args.check_only:
            print(f'Preflight passed: {result["changed_count"]} changed tensors; no model written')
        else:
            if output.exists():
                raise ValueError('Output appeared during conversion; candidate kept in staging')
            (job / 'candidate').rename(output)
            print(f'Candidate saved: {output}; runtime unverified')
            shutil.rmtree(job)
    finally:
        if args.check_only:
            shutil.rmtree(job)


def extract_expert_source(source, records, destination, check_only):
    from build import read_header
    index = json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    plan = []
    for record in records:
        layer = int(record['name'].split('.')[1])
        name = f'model.language_model.layers.{layer}.mlp.experts.down_proj'
        shard = index.get(name)
        if not shard or Path(shard).name != shard:
            raise ValueError('Missing fused BF16 expert: '+name)
        file = source/shard
        header, start = read_header(file)
        tensor = header[name]
        begin,end = tensor['data_offsets']
        if tensor['dtype']!='BF16' or tensor['shape']!=[512,2560,640] or end-begin!=512*2560*640*2 or file.stat().st_size<start+end:
            raise ValueError('Unsupported BF16 expert layout: '+name)
        plan.append((layer,file,start+begin,end-begin))
    if check_only:
        return None
    if shutil.disk_usage(destination.parent).free < sum(size for _,_,_,size in plan)+(1<<30):
        raise ValueError('Insufficient space for local BF16 expert extraction')
    destination.mkdir()
    status={'status':'complete','tensors':{}}
    for layer,file,offset,size in plan:
        digest=hashlib.sha256()
        with file.open('rb') as inp,(destination/f'layer-{layer:02d}-down.bf16').open('xb') as out:
            inp.seek(offset)
            remaining=size
            while remaining:
                block=inp.read(min(8<<20,remaining))
                if not block:raise ValueError('Truncated BF16 expert source')
                out.write(block);digest.update(block);remaining-=len(block)
        status['tensors'][str(layer)]={'sha256':digest.hexdigest()}
    (destination/'status.json').write_text(json.dumps(status))
    (destination/'source-manifest.json').write_text(json.dumps({'path':str(source),'method':'local safetensors range extraction'}))
    return destination


def execute(request_path):
    # Imported only in the offline helper, which already contains torch/ModelOpt.
    from audit import audit
    from build import build, mapped_name, read_header
    from verify import verify
    request = json.loads(request_path.read_text())
    job = request_path.parent
    original, donor, base = (Path(request[k]) for k in ('original', 'donor', 'base'))
    cache = Path(request['cache'])
    audit_file = job / 'audit.json'
    audit(cache / 'hub', audit_file, original_dir=original, donor_dir=donor)
    report = json.loads(audit_file.read_text())
    changes = [r for r in report['tensors'] if r['changed']]
    experts = []
    weight_map = json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
    headers = {}
    for record in changes:
        if record['name'].endswith('.ffn_down_exps.weight'):
            if record['type'] != 'Q8_0' or record['shape'] != [640, 2560, 512]:
                raise ValueError('Unsupported Qwen expert layout')
            experts.append(record)
        else:
            name = mapped_name(record)
            shard = weight_map[name]
            if shard not in headers: headers[shard] = read_header(base/shard)[0]
            tensor = headers[shard][name]
            if tensor['dtype'] != 'BF16' or tensor['shape'] != list(reversed(record['shape'])):
                raise ValueError('Base tensor does not match GGUF layout: '+name)
    raw_source = Path(request['source_raw']) if request.get('source_raw') else None
    if experts and request.get('source_bf16'):
        raw_source = extract_expert_source(Path(request['source_bf16']), experts, job/'expert-source', request['check_only'])
    if experts and not (request.get('source_bf16') and request['check_only']):
        if raw_source is None:
            raise ValueError('Changed experts require --source-bf16 with the original BF16 model (or --source-raw slices)')
        status = json.loads((raw_source / 'status.json').read_text())
        if status.get('status') != 'complete' or not (raw_source / 'source-manifest.json').is_file():
            raise ValueError('BF16 expert source is incomplete')
        for record in experts:
            layer = int(record['name'].split('.')[1])
            file = raw_source / f'layer-{layer:02d}-down.bf16'
            if file.stat().st_size != 512 * 2560 * 640 * 2:
                raise ValueError(f'Invalid BF16 expert size: {file}')
            with file.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            if checksum != status['tensors'][str(layer)]['sha256']:
                raise ValueError(f'BF16 expert checksum mismatch: {file}')
    hashes = {}
    index = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    for name in sorted(set(index.values())):
        with (base / name).open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not request['check_only']:
        size = sum((base / name).stat().st_size for name in hashes)
        if shutil.disk_usage(job).free < size + (1 << 30):
            raise ValueError('Insufficient space for an independent candidate copy')
        candidate = job / 'candidate'
        build(cache, audit_file, candidate, '', source=base, independent=True)
        if experts:
            from patch_experts import run as patch
            patch(cache, audit_file, raw_source, candidate, source=base)
        verification = job / 'verification.json'
        verify(cache, candidate, verification, source=base, source_hashes=hashes)
        shutil.copyfile(audit_file, candidate / 'gguf-audit.json')
        shutil.copyfile(verification, candidate / 'verification.json')
        manifest = {'status': 'candidate_verified', 'profile': 'qwen38_huihui',
                    'runtime_validated': False, 'activation_scale_policy': 'preserve',
                    'paths': {k: request[k] for k in ('original', 'donor', 'base')},
                    'changed_count': len(changes), 'source_shard_hashes': hashes,
                    'verification': json.loads(verification.read_text())}
        (candidate / 'conversion-manifest.json').write_text(json.dumps(manifest, indent=2))
    (job / 'result.json').write_text(json.dumps({'status': 'passed', 'changed_count': len(changes)}))


if __name__ == '__main__':
    execute(Path(sys.argv[1]))
