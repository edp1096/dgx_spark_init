"""Three local checkpoints -> verified candidate. No hub clients or subprocesses."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .numeric import bf16_to_float
from .planning import audit, load_profile
from model_adapters.fused_experts import aligned_checkpoints
from .quantization import fp8_decode, quantize_fp8, quantize_nvfp4
from .safetensors_io import Checkpoint, replace_shard

# Runtime metadata only; upstream benchmark claims and README are not inherited.
METADATA = {'config.json', 'hf_quant_config.json', 'generation_config.json',
            'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json',
            'vocab.json', 'merges.txt', 'chat_template.jinja', 'preprocessor_config.json',
            'processor_config.json', 'video_preprocessor_config.json',
            'special_tokens_map.json', 'added_tokens.json', 'LICENSE', 'LICENSE.txt'}


def scale_key(base, weight, suffix):
    prefix = weight.removesuffix('.weight')
    candidates = [prefix + '.' + suffix, weight + '_' + suffix.removeprefix('weight_')]
    if suffix == 'weight_scale':
        candidates += [weight+'_scale_inv', weight+'_scale', weight+'.weight_scale', weight+'_weight_scale']
    matches = list(dict.fromkeys(k for k in candidates if k in base.tensors))
    if len(matches) != 1:
        raise ValueError(f'Expected one {suffix} for {weight}: {matches}')
    return matches[0]


def require_layout(base, key, dtype, shape):
    target = base.tensors[key]
    if target['dtype'] != dtype or target['shape'] != list(shape):
        raise ValueError(f'Unsupported destination layout: {key} {target["dtype"]} {target["shape"]}')


def fp8_block_layout(weight_shape, scale_shape, quant):
    if len(weight_shape)!=2:raise ValueError('FP8 requires 2D weights')
    if len(scale_shape)==4:
        if scale_shape[1]!=1 or scale_shape[3]!=1:raise ValueError('Invalid 4D FP8 scale layout')
        grid=[scale_shape[0],scale_shape[2]]
    elif len(scale_shape)==2 and scale_shape != [weight_shape[0],1]:
        grid=scale_shape
    else:return None
    block=quant.get('weight_block_size')
    if block is None:
        # Match the legacy square layout only when the dimensions determine it.
        if any(w%s for w,s in zip(weight_shape,grid)):raise ValueError('Partial FP8 blocks require weight_block_size metadata')
        block=[w//s for w,s in zip(weight_shape,grid)]
        if block[0]!=block[1]:raise ValueError('Ambiguous FP8 block layout; set weight_block_size metadata')
    if len(block)!=2 or any(type(v)!=int or v<1 for v in block):raise ValueError('Invalid FP8 block size')
    if [(w+b-1)//b for w,b in zip(weight_shape,block)]!=grid:raise ValueError('FP8 scale grid does not match block size')
    return block


def expand_fp8_scale(scale, shape, block):
    if block is not None:
        br,bc=block;rows,cols=shape
        grid=scale.reshape((rows+br-1)//br,(cols+bc-1)//bc)
        return np.repeat(np.repeat(grid,br,axis=0),bc,axis=1)[:rows,:cols]
    return scale if scale.size==1 else scale[:,None]


def prepare(original, donor, base, profile, activation_scales, max_tensor_bytes):
    report = audit(original, donor, base, profile)
    base = Checkpoint(base)
    quant = report['base_quantization_config'] or {}
    plans = []
    for change in report['changes']:
        name = change['name']
        if change['action'] == 'blocked':
            raise ValueError(f'{name}: {change["reason"]}')
        if np.prod(change['shape'], dtype=np.int64) * (4 if change['source_dtype']=='F32' else 2) > max_tensor_bytes:
            raise ValueError(f'{name}: exceeds per-tensor memory limit; explicit slicing adapter required')
        plan = {'name': name, 'kind': 'raw' if change['action']=='copy_donor_raw' else 'bf16', 'scales': []}
        if change['action'] == 'requires_quantizer':
            legacy_fp8 = report['profile_config'].get('fp8_transfer') == 'base_plus_delta'
            if not legacy_fp8 and quant.get('quant_method') != 'modelopt':
                raise ValueError('Only explicit ModelOpt quantized layouts are supported')
            if not legacy_fp8 and quant.get('quant_algo') not in ('NVFP4', 'FP8', 'MIXED_PRECISION'):
                raise ValueError('Unsupported ModelOpt quant_algo')
            prefix = name.removesuffix('.weight')
            unsupported = [key for key in base.tensors if key.startswith(prefix+'.') and
                           ('pre_quant_scale' in key or 'awq' in key.lower())]
            if unsupported:
                raise ValueError(f'Preconditioned weights need a dedicated adapter: {unsupported}')
            dtype = base.tensors[name]['dtype']
            scale = scale_key(base, name, 'weight_scale')
            if dtype == 'F8_E4M3':
                require_layout(base, name, 'F8_E4M3', change['shape'])
                if base.tensors[scale]['dtype'] != 'F32':
                    raise ValueError('FP8 weight scale must be FP32')
                shape = base.tensors[scale]['shape']
                block = fp8_block_layout(change['shape'], shape, quant) if legacy_fp8 else None
                if block is None and np.prod(shape, dtype=np.int64) != 1 and shape not in ([change['shape'][0]], [change['shape'][0], 1]):
                    raise ValueError('Unsupported FP8 scale layout')
                plan.update(kind='fp8', scales=[scale], block_size=block, base_plus_delta=legacy_fp8)
            elif not legacy_fp8 and dtype == 'U8' and quant['quant_algo'] in ('NVFP4', 'MIXED_PRECISION'):
                if len(change['shape']) != 2 or change['shape'][1] % 16:
                    raise ValueError('Unsupported NVFP4 source layout')
                rows, cols = change['shape']
                require_layout(base, name, 'U8', [rows, cols // 2])
                require_layout(base, scale, 'F8_E4M3', [rows, cols // 16])
                global_scale = scale_key(base, name, 'weight_scale_2')
                if base.tensors[global_scale]['dtype'] != 'F32' or np.prod(base.tensors[global_scale]['shape'], dtype=np.int64) != 1:
                    raise ValueError('NVFP4 global scale must be scalar FP32')
                plan.update(kind='nvfp4', scales=[scale, global_scale])
            else:
                raise ValueError(f'Unsupported packed dtype: {dtype}')
            # Static input scales can change after upstream weights change. Never
            # silently claim activation calibration from just three checkpoints.
            if activation_scales != 'preserve':
                raise ValueError('Quantized changes require --activation-scales preserve; fresh calibration is not implemented')
        plans.append(plan)
    claimed = [key for p in plans for key in [p['name'], *p['scales']]]
    if len(claimed) != len(set(claimed)):
        raise ValueError('Overlapping weight/scale replacements')
    return report, plans


def verify_quantized_origin(original_cp, base_cp, plan, max_relative_error):
    name = plan['name']
    source = bf16_to_float(b''.join(original_cp.blocks(name))).reshape(original_cp.tensors[name]['shape'])
    if not np.isfinite(source).all():
        raise ValueError(f'Non-finite original: {name}')
    codes = np.frombuffer(b''.join(base_cp.blocks(name)), dtype=np.uint8).reshape(base_cp.tensors[name]['shape'])
    scales = plan['scales']
    if plan['kind'] == 'fp8':
        scale = np.frombuffer(b''.join(base_cp.blocks(scales[0])), dtype='<f4')
        if np.any(scale <= 0) or not np.isfinite(scale).all():
            raise ValueError('Invalid original FP8 scale')
        reconstructed = fp8_decode(codes) * expand_fp8_scale(scale, source.shape, plan.get('block_size'))
    else:
        scale = fp8_decode(np.frombuffer(b''.join(base_cp.blocks(scales[0])), dtype=np.uint8)).reshape(base_cp.tensors[scales[0]]['shape'])
        global_scale = np.frombuffer(b''.join(base_cp.blocks(scales[1])), dtype='<f4')[0]
        if np.any(scale <= 0) or not np.isfinite(global_scale) or global_scale <= 0:
            raise ValueError('Invalid original NVFP4 scale')
        unpacked = np.empty(source.shape, dtype=np.uint8)
        unpacked[:, ::2], unpacked[:, 1::2] = codes & 15, codes >> 4
        levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
        reconstructed = levels[unpacked & 7] * np.where(unpacked & 8, -1, 1)
        reconstructed *= np.repeat(scale * global_scale, 16, axis=1)
    error = float(np.linalg.norm(reconstructed.astype(np.float64)-source)) / max(float(np.linalg.norm(source.astype(np.float64))), 1e-30)
    if not np.isfinite(error) or error > max_relative_error:
        raise ValueError(f'Quantized base does not align with original: {name}: {error}')
    return error


def convert(original, donor, base, profile, output, activation_scales='reject',
            max_tensor_bytes=128*1024*1024, max_relative_error=0.3):
    output = Path(output).absolute()
    inputs = [Path(p).resolve(strict=True) for p in (original, donor, base)]
    resolved_output = output.resolve()
    if any(resolved_output == p or p in resolved_output.parents or resolved_output in p.parents for p in inputs):
        raise ValueError('Output must be separate from all input directories')
    if output.exists():
        raise FileExistsError(output)
    if not 0 < max_relative_error < 1 or max_tensor_bytes <= 0:
        raise ValueError('Invalid numeric limits')
    report, plans = prepare(*inputs, profile, activation_scales, max_tensor_bytes)
    original_cp, donor_cp, base_cp = aligned_checkpoints(*inputs,load_profile(profile))
    alignment = {p['name']: verify_quantized_origin(original_cp, base_cp, p, max_relative_error)
                 for p in plans if p['kind'] in ('fp8','nvfp4')}
    # Hash all base tensors for preservation checks and source mutation detection.
    base_hashes = {key: base_cp.digest(key) for key in base_cp.tensors}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+output.name+'.', dir=output.parent))
    reserved = False
    try:
        payload_dir = stage / '.payloads'
        payload_dir.mkdir()
        replacements, expected = {}, {}
        metrics = []
        changed = {v['name']: v for v in report['changes']}
        for plan in plans:
            name = plan['name']
            if donor_cp.digest(name) != changed[name]['donor_sha256']:
                raise ValueError(f'Donor changed during conversion: {name}')
            raw = b''.join(donor_cp.blocks(name))
            if hashlib.sha256(raw).hexdigest() != changed[name]['donor_sha256']:
                raise ValueError(f'Donor read changed: {name}')
            values = (np.frombuffer(raw,dtype={'F16':'<f2','F32':'<f4'}[changed[name]['source_dtype']]).astype(np.float32) if plan['kind']=='raw' else bf16_to_float(raw)).reshape(changed[name]['shape'])
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite donor: {name}')
            if plan['kind'] in ('bf16','raw'):
                encoded, reconstructed = [raw], values
            elif plan['kind'] == 'fp8':
                if plan.get('base_plus_delta'):
                    origin_raw = b''.join(original_cp.blocks(name))
                    if hashlib.sha256(origin_raw).hexdigest() != changed[name]['original_sha256']:
                        raise ValueError('Original changed during conversion: '+name)
                    origin = bf16_to_float(origin_raw).reshape(values.shape)
                    base_codes = np.frombuffer(b''.join(base_cp.blocks(name)), dtype=np.uint8).reshape(values.shape)
                    base_scale = np.frombuffer(b''.join(base_cp.blocks(plan['scales'][0])), dtype='<f4')
                    values = fp8_decode(base_codes).astype(np.float32)*expand_fp8_scale(base_scale, values.shape, plan.get('block_size')) + (values-origin)
                codes, scale, reconstructed = quantize_fp8(values, base_cp.tensors[plan['scales'][0]]['shape'], plan.get('block_size'), 1e-12 if plan.get('base_plus_delta') else 1e-30)
                encoded = [codes.tobytes(), scale.astype('<f4').tobytes()]
            else:
                codes, scale, global_scale, reconstructed = quantize_nvfp4(values)
                encoded = [codes.tobytes(), scale.tobytes(), global_scale.tobytes()]
            denominator = float(np.linalg.norm(values.astype(np.float64)))
            error = float(np.linalg.norm(reconstructed.astype(np.float64)-values)) / max(denominator, 1e-30)
            if not np.isfinite(error) or error > max_relative_error:
                raise ValueError(f'Quantization error too large: {name}: {error}')
            metrics.append({'name': name, 'kind': plan['kind'], 'relative_l2_error': error})
            for key, data in zip([name, *plan['scales']], encoded, strict=True):
                file = payload_dir / str(len(expected))
                file.write_bytes(data)
                expected[key] = hashlib.sha256(data).hexdigest()
                shard = base_cp.tensors[key]['shard']
                replacements.setdefault(shard, {})[key] = (file, 0, len(data))
        for shard in sorted({t['shard'] for t in base_cp.tensors.values()}):
            if shard in replacements:
                replace_shard(inputs[2]/shard, stage/shard, replacements[shard])
            else:
                shutil.copyfile(inputs[2]/shard, stage/shard)
        for name in METADATA:
            if (inputs[2]/name).is_file():
                shutil.copyfile(inputs[2]/name, stage/name)
        result = Checkpoint(stage)
        for name in base_cp.tensors:
            if result.digest(name) != expected.get(name, base_hashes[name]):
                raise ValueError(f'Output verification failed: {name}')
            if base_cp.digest(name) != base_hashes[name]:
                raise ValueError(f'Base changed during conversion: {name}')
        for change in report['changes']:
            name = change['name']
            if original_cp.digest(name) != change['original_sha256'] or donor_cp.digest(name) != change['donor_sha256']:
                raise ValueError(f'Source weights changed during conversion: {name}')
        shutil.rmtree(payload_dir)
        report.update(status='candidate_verified', conversion_ready=True,
                      note='Candidate passed tensor preservation and numeric checks; runtime and activation calibration remain unverified',
                      runtime_validated=False, activation_calibration='not performed',
                      activation_scale_policy=activation_scales,
                      quantizer='CPU reference: E4M3 / ModelOpt-layout NVFP4, weight amax, no AWQ/GPTQ',
                      metrics=metrics, original_alignment_relative_errors=alignment, output_hashes=expected,
                      preserved_base_tensors=len(base_hashes)-len(expected))
        (stage/'conversion-manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        # Reserve destination exclusively; rename over our empty directory only.
        output.mkdir()
        reserved = True
        os.replace(stage, output)
        reserved = False
        return report
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if reserved:
            output.rmdir()
