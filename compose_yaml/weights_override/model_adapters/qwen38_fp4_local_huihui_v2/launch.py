"""Launch an isolated TP1 GPU qualification server for FP4-Local."""
import argparse
import json
from pathlib import Path
import socket
import subprocess

from download import atomic, digest

IMAGE = 'local/qwen38-fp4-local:csf-yarn-20261016'
BASE_DIGEST = 'sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595'


def launch(root, model, label, context, port, utilization, dry_run=False):
    model = model.resolve(strict=True)
    config = json.loads((model / 'config.json').read_text())
    if config['quantization_config']['quant_algo'] != 'MIXED_PRECISION':
        raise ValueError('Expected FP4-Local mixed precision')
    if not dry_run:
        receipt_name = 'download-receipt.json' if label.startswith('base') else 'transfer-manifest.json'
        receipt = json.loads((model / receipt_name).read_text())
        status = 'verified' if label.startswith('base') else 'candidate_verified'
        revision = receipt.get('revision') if label.startswith('base') else receipt.get('base_revision')
        if receipt['status'] != status or revision != 'b9ce18a6887e18c089fdea8509a33e63429a3300':
            raise ValueError('A completely verified checkpoint is required')
    if not 0.7 <= utilization <= .9:
        raise ValueError('Unexpected memory utilization')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    name = 'qwen38-fp4-local-' + label
    if subprocess.run(['docker', 'inspect', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise FileExistsError('Qualification container already exists: ' + name)
    output = root / 'runtime' / label
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'cache'
    cache.mkdir(exist_ok=True)
    triton_cache = cache / 'triton'
    triton_cache.mkdir(exist_ok=True)
    if context > 262144:
        if context != 1048576:
            raise ValueError('Only the explicit factor-4 YaRN extension is supported')
        text = config['text_config']
        text['max_position_embeddings'] = context
        text['rope_parameters'] = {**text['rope_parameters'], 'rope_type': 'yarn',
            'factor': 4., 'original_max_position_embeddings': 262144}
    overlay = output / 'runtime-config.json'
    atomic(overlay, config)
    env = {
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'VLLM_NO_USAGE_STATS': '1',
        'VLLM_PLE_TABLE_MEMORY': 'disk', 'B12X_DISK_BACKEND': 'io_uring',
        'CUTE_DSL_ARCH': 'sm_121a', 'SAFETENSORS_FAST_GPU': '1',
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_SSM_CONV_STATE_LAYOUT': 'DS',
        'VLLM_USE_AOT_COMPILE': '1', 'VLLM_USE_MEGA_AOT_ARTIFACT': '1',
        'VLLM_USE_V2_MODEL_RUNNER': '1', 'B12X_POLICY_MODE': 'auto',
        'VLLM_MXFP8_LM_HEAD': '1', 'VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE': 'w13',
        'VLLM_B12X_MOE_FP4_FORCE_A16': '1', 'B12X_W4A16_A4_PREFILL': '1',
        'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '8',
    }
    served = 'nvidia/Qwen3.8-Flash-Next-FP4-Local' if label.startswith('base') else 'edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local'
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', IMAGE]))[0]
    labels = image['Config'].get('Labels') or {}
    if labels.get('sparktalk.fp4-local.yarn-config') != '2' or labels.get('sparktalk.fp4-local.base-digest') != BASE_DIGEST:
        raise ValueError('Unexpected qualification runtime image')
    if context > 262144 and not dry_run:
        check = json.loads((root / 'runtime/rope-qualification.json').read_text())
        if check['status'] != 'passed' or check['image_id'] != image['Id'] or check['config_sha256'] != digest(overlay):
            raise ValueError('This exact runtime/long-context configuration needs a RoPE reference check')
    command = ['docker', 'run', '-d', '--name', name, '--gpus', 'all', '--init', '--ipc=host',
        '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864', '--security-opt', 'seccomp=unconfined',
        '-p', f'127.0.0.1:{port}:8000', '-v', f'{model}:/model:ro',
        '-v', f'{overlay}:/model/config.json:ro', '-v', f'{cache}:/root/.cache',
        '-v', f'{triton_cache}:/root/.triton',
        '--entrypoint', 'vllm']
    for key, value in env.items():
        command += ['-e', f'{key}={value}']
    command += [image['Id'], 'serve', '/model', '--served-model-name', served,
        '--host', '0.0.0.0', '--port', '8000', '--tensor-parallel-size', '1', '--pipeline-parallel-size', '1',
        '--mamba-cache-mode', 'align', '--enable-prefix-caching', '--enable-chunked-prefill',
        '--dtype', 'bfloat16', '--kv-cache-dtype', 'fp8', '--quantization', 'modelopt_mixed',
        '--block-size', '16', '--load-format', 'b12x', '--max-model-len', str(context),
        '--max-num-seqs', '1', '--max-num-batched-tokens', '4096',
        '--speculative-config', '{"method":"mtp","num_speculative_tokens":4}',
        '--gdn-decode-kernel', 'b12x', '--linear-backend', 'b12x', '--moe-backend', 'b12x',
        '--no-enable-flashinfer-autotune', '--mm-encoder-tp-mode', 'data',
        '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_xml', '--enable-auto-tool-choice',
        '--compilation-config', '{"pass_config":{"fuse_act_quant":true}}',
        '--gpu-memory-utilization', str(utilization)]
    record = {'container': name, 'image_id': image['Id'], 'base_image_digest': BASE_DIGEST,
              'model': str(model), 'served_model': served, 'context': context,
              'runtime_config_sha256': digest(overlay), 'original_config_sha256': digest(model / 'config.json'),
              'port': port, 'command': command, 'environment': env,
              'meminfo_before': Path('/proc/meminfo').read_text()}
    if dry_run:
        atomic(output / 'launch-plan.json', record)
        print('Launch configuration prepared without starting a server', flush=True)
        return record
    atomic(output / 'launch.json', record)
    record['container_id'] = subprocess.check_output(command, text=True).strip()
    atomic(output / 'launch.json', record)
    print(name, record['container_id'], flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--label', choices=['base', 'candidate', 'base-native', 'candidate-native'], required=True)
    parser.add_argument('--context', type=int, choices=[262144, 1048576], default=1048576)
    parser.add_argument('--port', type=int, default=30136)
    parser.add_argument('--utilization', type=float, default=.8)
    parser.add_argument('--dry-run', action='store_true')
    a = parser.parse_args()
    launch(a.root, a.model, a.label, a.context, a.port, a.utilization, a.dry_run)
