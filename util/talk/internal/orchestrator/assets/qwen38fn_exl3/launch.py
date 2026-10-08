"""Create a YaRN model view; original checkpoint files remain read-only."""
import json
import os
from pathlib import Path
import sys
import yaml

REPO = 'alesha-pro/Huihui-Qwen3.8-Flash-Next-abliterated-exl3-3bit-hq_h6_ng6'
REVISION = '3b585c458f9fcf3322e76cff2c635c2cb81c5869'
MODEL = 'qwen38fn_exl3'
TRAINED_CONTEXT = 262144


def prepare_runtime(source, runtime, limit=1048576):
    if limit not in (262144, 524288, 1048576):
        raise ValueError('Qwen 3.8 Flash-Next EXL3 supports the tested 262K, 512K and 1M windows')
    config = json.loads((source / 'config.json').read_text())
    required = [f'model-{i:05d}-of-00007.safetensors' for i in range(1, 8)] + ['ngram_embedding.safetensors', 'tokenizer.json']
    if any(not (source / name).is_file() for name in required):
        raise RuntimeError('Original Huihui EXL3 checkpoint is incomplete; run model preparation')
    text_config = config['text_config']
    rope = text_config['rope_parameters']
    if limit != TRAINED_CONTEXT:
        config['max_position_embeddings'] = limit
        text_config['max_position_embeddings'] = limit
        rope.update(rope_type='yarn', factor=limit / TRAINED_CONTEXT,
                    original_max_position_embeddings=TRAINED_CONTEXT)
    models = runtime / 'models'
    view = models / MODEL
    view.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name == 'config.json' or not path.is_file():
            continue
        target = view / path.name
        if target.is_symlink():
            if target.readlink() == path:
                continue
            target.unlink()
        elif target.exists():
            raise RuntimeError(f'Model view contains an unexpected regular file: {target.name}')
        target.symlink_to(path)
    temporary = view / 'config.json.tmp'
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    temporary.replace(view / 'config.json')
    serving = {
        'network': {'host': '0.0.0.0', 'port': 30000, 'disable_auth': True, 'allowed_origins': []},
        'logging': {'log_prompt': False, 'log_generation_params': True, 'log_timestamps': True},
        'model': {'model_dir': str(models), 'model_name': MODEL, 'backend': 'exllamav3',
                  'max_seq_len': limit, 'cache_size': limit, 'cache_mode': 'Q8',
                  'chunk_size': 2048, 'max_batch_size': 1, 'ngram_ram': False,
                  'vision': True, 'vision_offload': False, 'warmup': True,
                  'tool_format': 'qwen3_5', 'gpu_split_auto': True},
        'draft_model': {'draft_mode': 'mtp', 'draft_cache_mode': 'Q8',
                        'draft_num_tokens': 3, 'dynamic_draft': False},
        'memory': {'sysmem_recurrent_cache': 4096, 'sysmem_kv_cache': 0,
                   'sysmem_multimodal_cache': 256},
    }
    config_path = runtime / 'config.yml'
    config_path.write_text(yaml.safe_dump(serving, sort_keys=False))
    return config_path


if __name__ == '__main__':
    source = Path('/hf/hub') / ('models--' + REPO.replace('/', '--')) / 'snapshots' / REVISION
    config_path = prepare_runtime(source, Path('/runtime'))
    print('SPARKTALK_QWEN38FN_EXL3_CONFIG context=1048576 cache=1048576 yarn=4 mtp=3 vision=1 ngram=disk', flush=True)
    os.chdir('/opt/tabbyAPI')
    os.execv(sys.executable, [sys.executable, '/opt/tabbyAPI/main.py', '--config', str(config_path)])
