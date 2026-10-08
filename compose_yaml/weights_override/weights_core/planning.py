"""Offline BF16 donor audit and quantized destination planning."""
import json
from pathlib import Path
from .safetensors_io import Checkpoint
from model_adapters.fused_experts import aligned_checkpoints

PROFILE_ROOT = Path(__file__).resolve().parents[1] / 'profiles'


def load_profile(name):
    path = PROFILE_ROOT / f'{name}.json' if name in {'ornith_15', 'gemma4_26b', 'fp8'} else Path(name)
    profile = json.loads(path.read_text())
    if not isinstance(profile.get('architecture'), str) or not isinstance(profile.get('text_config'), dict):
        raise ValueError('Profile requires architecture and text_config')
    if (not profile['text_config'] and not profile.get('match_source_structure')) or type(profile.get('runtime_tp')) is not int or profile['runtime_tp'] < 1:
        raise ValueError('Profile requires structural checks and positive runtime_tp')
    return profile


def check_config(checkpoint, profile):
    config = json.loads((checkpoint.root / 'config.json').read_text())
    if not profile.get('match_source_structure') and config.get('architectures') != [profile['architecture']]:
        raise ValueError(f'Architecture mismatch: {checkpoint.root}')
    text = config.get('text_config', config)
    for key, value in profile['text_config'].items():
        if text.get(key) != value:
            raise ValueError(f'{key} mismatch: {checkpoint.root}')
    return config


def audit(original, donor, base, profile_name):
    profile = load_profile(profile_name)
    checkpoints = aligned_checkpoints(original, donor, base, profile)
    configs = [check_config(cp, profile) for cp in checkpoints]
    if any(not c.get("architectures") for c in configs) or any(c["architectures"] != configs[0]["architectures"] for c in configs):
        raise ValueError("Input architectures differ or are missing")
    original, donor, base = checkpoints
    def structural(config):
        ignored = {'_name_or_path', 'torch_dtype', 'dtype', 'transformers_version', 'quantization_config'}
        if isinstance(config, dict):
            return {key: structural(value) for key, value in config.items() if key not in ignored}
        if isinstance(config, list):
            return [structural(value) for value in config]
        return config
    # A weights-only transfer cannot carry architecture/tokenizer changes.
    texts = [config.get('text_config', config) for config in configs]
    if structural(texts[0]) != structural(texts[1]) or structural(texts[0]) != structural(texts[2]):
        raise ValueError('Text model configs differ beyond dtype/quantization metadata')
    if original.tensors.keys() != donor.tensors.keys():
        raise ValueError('Original/donor tensor keys differ; explicit mapping required')
    changes = []
    preserved = 0
    for name, src in original.tensors.items():
        dst = donor.tensors[name]
        if (src['dtype'], src['shape']) != (dst['dtype'], dst['shape']):
            raise ValueError(f'Original/donor layout mismatch: {name}')
        before, after = original.digest(name), donor.digest(name)
        if before == after:
            preserved += 1
            continue
        item = {'name': name, 'shape': src['shape'], 'source_dtype': src['dtype'],
                'original_sha256': before, 'donor_sha256': after,
                'source_tensor': src.get('source_tensor',name), 'expert_index': src.get('expert_index')}
        target = base.tensors.get(name)
        if profile.get('fp8_transfer') == 'base_plus_delta' and src['dtype'] in ('F16','F32') and target is not None and target['dtype']==src['dtype'] and target['shape']==src['shape']:
            if base.digest(name)!=before:
                item.update(action='blocked',reason='Unquantized base differs from original')
            else:
                item.update(action='copy_donor_raw',base_sha256=before)
        elif src['dtype'] != 'BF16':
            item.update(action='blocked', reason='Changed donor tensor is not BF16')
        elif target is None:
            item.update(action='blocked', reason='Destination key missing; layout adapter required')
        elif target['dtype'] == 'BF16' and target['shape'] == src['shape']:
            base_hash = base.digest(name)
            if base_hash != before:
                item.update(action='blocked', reason='Base BF16 differs from original', base_sha256=base_hash)
            else:
                item.update(action='copy_donor_bf16', base_sha256=base_hash)
        else:
            # A packed UINT8 payload alone cannot establish its quantization format.
            # Require adapter/calibration validation rather than guessing from dtype.
            item.update(action='requires_quantizer', destination_dtype=target['dtype'],
                        destination_shape=target['shape'],
                        reason='Model-specific packing and weight/activation scales must be validated')
        changes.append(item)
    return {'status': 'audit_complete', 'conversion_ready': False,
            'profile': str(profile_name), 'profile_config': profile, 'runtime_tp': profile['runtime_tp'],
            'paths': {k: str(cp.root) for k, cp in zip(('original', 'donor', 'base'), checkpoints)},
            'base_quantization_config': configs[2].get('quantization_config'),
            'unchanged_source_tensors': preserved, 'changes': changes,
            'note': 'Read-only plan. Quantizer/calibration and TP1 runtime validation remain separate.'}
