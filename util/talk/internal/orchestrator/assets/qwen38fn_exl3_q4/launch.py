"""Qualified GPU EXL3 4bit profile; never downloads or converts model weights."""
import json
import os
from pathlib import Path

MODEL = Path('/hf/hub/models--alesha-pro--Huihui-Qwen3.8-Flash-Next-abliterated-exl3-4bit-hq_h6_ng6/snapshots/e884d3e5e38d53e3b50a59e02d1b7a4cb1e5d75e')
MODEL_ID = 'qwen38fn_exl3_q4'


def command():
    shards = sorted(MODEL.glob('model-?????-of-00009.safetensors'))
    if len(shards) != 9 or any(not p.is_file() for p in shards):
        raise RuntimeError('The pinned EXL3 4bit checkpoint is missing; prepare the model first')
    for name in ('ngram_embedding.safetensors', 'config.json', 'tokenizer.json'):
        if not (MODEL / name).is_file():
            raise RuntimeError(f'Missing pinned model file: {name}')
    return ['./gb10_inference', '--server', '--model-dir', str(MODEL),
            '--host', '0.0.0.0', '--port', '30000', '--model-name', MODEL_ID,
            '--max-seq-len', '1048576', '--rope-yarn-factor', '4', '--max-batch', '1',
            '--ple-ram', 'ssd', '--kv-cache', 'q8', '--exl3-mtp-k', '3',
            '--draft-confidence', '0', '--prefix-cache', 'on', '--prefix-ckpt-mem-gb', '4',
            '--prefill-chunk', '2048', '--tune-table', 'off', '--reasoning-effort', 'none',
            '--thinking', 'off', '--exl3-pdl', '0', '--exl3-mtp-head-n', '0']


if __name__ == '__main__':
    os.chdir('/opt/velogb10')
    Path('/runtime').mkdir(parents=True, exist_ok=True)
    args = command()
    Path('/runtime/launch.json').write_text(json.dumps(args))
    os.execv(args[0], args)
