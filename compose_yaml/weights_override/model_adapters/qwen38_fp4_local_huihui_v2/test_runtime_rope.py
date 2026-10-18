"""Check multimodal YaRN against ordinary YaRN at identical text positions."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.transformers_utils.config import get_config
from vllm.model_executor.layers.rotary_embedding import get_rope


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/model')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image-id', required=True)
    args = p.parse_args()
    config = get_config(args.model, trust_remote_code=False)
    text = config.get_text_config()
    if config.max_position_embeddings != 1048576 or text.max_position_embeddings != 1048576:
        raise AssertionError('Composite/text context lengths differ')
    reports = []
    with set_current_vllm_config(VllmConfig()):
        for name, maximum, parameters in [
            ('yarn_1m', 1048576, dict(text.rope_parameters)),
            ('native_262k', 262144, {k: v for k, v in text.rope_parameters.items()
                 if k not in ('factor', 'original_max_position_embeddings', 'rope_type')} | {'rope_type': 'default'}),
        ]:
            multi = get_rope(head_size=256, max_position=maximum, rope_parameters=parameters, dtype=torch.bfloat16)
            plain_parameters = {k: v for k, v in parameters.items() if k not in ('mrope_section', 'mrope_interleaved')}
            plain = get_rope(head_size=256, max_position=maximum, rope_parameters=plain_parameters, dtype=torch.bfloat16)
            positions = torch.tensor([0, 1, 4095, 65535, maximum - 1])
            difference = (multi.cos_sin_cache[positions].float() - plain.cos_sin_cache[positions].float()).abs().max().item()
            if difference != 0. or not torch.isfinite(multi.cos_sin_cache[positions]).all():
                raise AssertionError((name, 'MRoPE/reference phase mismatch', difference))
            if multi.mrope_section != [11, 11, 10] or not multi.mrope_interleaved:
                raise AssertionError('Multimodal rotary sections changed')
            reports.append({'name': name, 'maximum_text_position': maximum - 1,
                'max_reference_difference': difference,
                'multimodal_cache_positions': multi.cos_sin_cache.shape[0],
                'plain_cache_positions': plain.cos_sin_cache.shape[0],
                'parameters': parameters})
    report = {'status': 'passed', 'image_id': args.image_id, 'tests': reports,
              'config_sha256': hashlib.sha256((Path(args.model) / 'config.json').read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
