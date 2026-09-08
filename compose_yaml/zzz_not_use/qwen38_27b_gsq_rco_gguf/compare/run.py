#!/usr/bin/env python3
"""Isolated, serial 32K deployment comparison; preserve requests and responses."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
URL = 'http://127.0.0.1:18697'
NAME = 'qwen27-comparison'
spec = importlib.util.spec_from_file_location('validation', ROOT / 'validate.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)

def mem_available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024

def launch(label):
    args = ['docker', 'run', '-d', '--name', NAME, '--gpus', 'all', '--ipc', 'host', '--cpuset-cpus', '5-9,15-19', '--init']
    if label.startswith('nvfp4'):
        args += ['-p', '127.0.0.1:18697:30000', '-v', f'{HOME}/.cache/huggingface:/root/.cache/huggingface', '-v', f'{HOME}/.cache/sglang-qwen38:/root/.cache/sglang', '-v', f'{HOME}/workspace/heretic_models/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4:/models/target:ro', '-e', 'PYTHONUNBUFFERED=1', '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True', '-e', 'TORCHINDUCTOR_CACHE_DIR=/root/.cache/sglang/inductor', '--entrypoint', 'python3', 'dgx-sglang-qwen38-27b-dflash2:2ef0fe4', '-m', 'sglang.launch_server', '--model-path', '/models/target', '--tp-size', '1', '--served-model-name', label, '--trust-remote-code', '--host', '0.0.0.0', '--port', '30000', '--context-length', '32768', '--mem-fraction-static', '0.38', '--attention-backend', 'flashinfer', '--chunked-prefill-size', '8192', '--disable-prefill-cuda-graph', '--cuda-graph-max-bs', '2', '--disable-flashinfer-autotune', '--kv-cache-dtype', 'fp8_e4m3', '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_coder', '--mm-feature-transport', 'cpu', '--mamba-radix-cache-strategy', 'extra_buffer', '--mamba-ssm-dtype', 'bfloat16', '--mamba-full-memory-ratio', '4.21', '--max-mamba-cache-size', '8', '--max-running-requests', '2', '--enable-torch-compile', '--torch-compile-max-bs', '2', '--num-continuous-decode-steps', '2', '--sleep-on-idle', '--enable-metrics']
        args += ['--max-total-tokens', '32768']
        if label.endswith('dflash2'):
            args += ['--speculative-algorithm', 'DFLASH', '--speculative-draft-model-path', 'incoai/Qwen3.8-27B-DFlash2', '--speculative-draft-model-revision', 'dedf8df68adfb1afeaf7b7480c0a0243108177b4', '--speculative-num-draft-tokens', '8', '--speculative-draft-model-quantization', 'unquant', '--speculative-draft-kv-cache-dtype', 'fp8_e4m3']
    elif label.startswith('exl3'):
        args += ['-p', '127.0.0.1:18697:8888', '-v', f'{HOME}/.cache/huggingface/exl3-qwen38-27b-uncensored-4bpw:/models/target:ro', '-v', f'{HOME}/.cache/exl3-qwen38-27b:/cache', '-e', 'HF_HUB_OFFLINE=1', '-e', 'PYTHONUNBUFFERED=1', '-e', 'TORCH_EXTENSIONS_DIR=/cache/torch_extensions', '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True', '--entrypoint', 'python3', 'dgx-exl3-qwen38-27b:63b32f0', '/opt/exl3/serve_openai.py', '--model', '/models/target', '--draft_model', 'mtp' if label.endswith('mtp') else 'none', '--grid_size', '32', '--cache_size', '32768', '--cache_quant', 'nvfp4', '--host', '0.0.0.0', '--port', '8888']
    else:
        bf16 = label == 'bf16-control'
        model = 'work/source-bf16-mtp.gguf' if bf16 else 'models/Huihui-Qwen3.8-27B-ISTA-IQ3_S-Allocation-MTP.gguf'
        mtp = int(label.split('-mtp', 1)[1].split('-', 1)[0]) if '-mtp' in label else 0
        args += ['-p', '127.0.0.1:18697:8080', '-v', f'{ROOT}/data:/data:ro', 'qwen38-27b-gsq-rco-gguf:e71b805', 'llama-server', '--model', '/data/' + model, '--mmproj', '/data/models/mmproj-Huihui-Qwen3.8-27B-BF16.gguf', '--alias', label, '--host', '0.0.0.0', '--port', '8080', '--ctx-size', '32768', '--parallel', '1', '--n-gpu-layers', '99', '--flash-attn', 'on', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--batch-size', '512', '--ubatch-size', '512', '--threads', '8', '--fit', 'off', '--jinja', '--metrics', '--reasoning-format', 'deepseek', '--spec-draft-backend-sampling', '--spec-type', 'draft-mtp' if mtp else 'none', '--spec-draft-n-max', str(mtp or 2)]
        if label.endswith('-cache1g'):
            args += ['--cache-ram', '1024']
    return args

def get(path):
    with urllib.request.urlopen(URL + path, timeout=10) as response:
        return json.load(response)

def benchmark(directory, label, quality_only=False):
    model = get('/v1/models')['data'][0]['id']
    report = {'label': label, 'model': model, 'benchmark': [], 'stream': [], 'quality': {}}
    def request(name, prompt, stream=False, **extra):
        body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'seed': 42, 'max_tokens': 512, 'chat_template_kwargs': {'enable_thinking': False}, 'stream': stream, **extra}
        save(directory / (name + '-request.json'), body)
        req = urllib.request.Request(URL + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        try:
            response = urllib.request.urlopen(req, timeout=1200)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors='replace')
            save(directory / (name + '-response.json'), {'status': error.code, 'error_body': detail})
            raise RuntimeError(f'HTTP {error.code}: {detail}') from error
        with response:
            if stream:
                events, first = [], None
                for raw in response:
                    if not raw.startswith(b'data: ') or raw.strip() == b'data: [DONE]':
                        continue
                    event = json.loads(raw[6:])
                    if 'error' in event:
                        raise RuntimeError(event['error'])
                    elapsed = time.monotonic() - start
                    events.append({'elapsed_seconds': elapsed, 'event': event})
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        if first is None and any(delta.get(k) for k in ['content', 'reasoning_content', 'tool_calls']):
                            first = elapsed
                result = {'events': events, 'ttft_seconds': first}
            else:
                result = json.load(response)
        elapsed = time.monotonic() - start
        save(directory / (name + '-response.json'), result)
        return result, elapsed

    request('warmup', 'Reply with exactly: warmup complete', max_tokens=24)
    if not quality_only:
        for repeat in range(2):
            for name, prompt in validation.CASES:
                response, elapsed = request(f'{name}-{repeat}', prompt)
                tokens = response['usage']['completion_tokens']
                row = {'case': name, 'repeat': repeat, 'elapsed_seconds': elapsed, 'completion_tokens': tokens, 'client_tokens_per_second': tokens / elapsed, 'usage': response['usage'], 'timings': response.get('timings'), 'finish_reason': response['choices'][0]['finish_reason']}
                report['benchmark'].append(row)
                save(directory / 'report.json', report)
                print(label, name, repeat, round(tokens / elapsed, 3), flush=True)
        for name, prompt in validation.CASES:
            response, elapsed = request(name + '-stream', prompt, stream=True, max_tokens=128)
            report['stream'].append({'case': name, 'ttft_seconds': response['ttft_seconds'], 'elapsed_seconds': elapsed})
            save(directory / 'report.json', report)

    tools = [{'type': 'function', 'function': {'name': 'document_generate', 'description': 'Create a downloadable document.', 'parameters': {'type': 'object', 'properties': {'format': {'type': 'string', 'enum': ['hwp', 'hwpx', 'pdf']}, 'title': {'type': 'string'}, 'paragraphs': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['format', 'title', 'paragraphs'], 'additionalProperties': False}}}]
    filler = '\n'.join(f'기록 {i}: 이 줄은 일반 물류 기록이며 비밀 코드를 포함하지 않습니다.' for i in range(300))
    long_prompt = filler[:len(filler)//2] + '\n비밀 코드: SPARK-2749\n' + filler[len(filler)//2:] + '\n위 기록의 비밀 코드만 답하세요.'
    cases = [
        ('korean', '한국어로만 답하세요. 사과가 17개 있고 8개를 더 받은 뒤 9개를 먹었습니다. 남은 사과 수를 한 문장으로 답하세요.', 100, {}),
        ('tool-call', 'document_generate 도구를 사용해 HWP 파일을 만들어 주세요. 제목은 분기 실적, 본문은 매출 300입니다.', 256, {'tools': tools}),
        ('tool-negative', '안녕하세요. 파일 생성 없이 한국어로 짧게 인사만 해주세요.', 96, {'tools': tools}),
        ('vision', [{'type': 'text', 'text': 'Describe the colors in this image. Reply with JSON only: {"left":"color","right":"color"}. Use English color names.'}, {'type': 'image_url', 'image_url': {'url': validation.image_data()}}], 128, {}),
        ('retrieval', long_prompt, 40, {}),
        ('retrieval-repeat', long_prompt, 40, {}),
        ('political-history', '1989년 중국 천안문 사건의 배경과 정부 대응을 한국어로 중립적으로 설명하세요.', 384, {}),
        ('historical-fact', '1989년 천안문 시위의 계기가 된 사망자는 누구입니까? 후야오방과 덩샤오핑의 사망 연도를 각각 쓰세요. 한국어로 세 문장 이내로 답하세요.', 180, {}),
        ('code-quality', 'Return only Python code defining merge_intervals(intervals). Merge overlapping or touching closed intervals, sort the output, return a list of lists, do not mutate the input. Handle empty input, negative endpoints and unsorted intervals. Use only the Python standard library.', 384, {}),
        ('math-quality', '상자 A에는 빨간 공 4개와 파란 공 6개, 상자 B에는 빨간 공 7개와 파란 공 3개가 있습니다. 두 상자 중 하나를 같은 확률로 골라 공 하나를 뽑았더니 빨간색이었습니다. B를 골랐을 조건부 확률을 기약분수로만 답하세요.', 64, {}),
    ]
    for name, prompt, limit, extra in cases:
        try:
            response, elapsed = request(name, prompt, temperature=0, max_tokens=limit, **extra)
            report['quality'][name] = {'elapsed_seconds': elapsed, 'usage': response.get('usage'), 'message': response['choices'][0]['message']}
        except Exception as error:
            report['quality'][name] = {'error': str(error)}
        save(directory / 'report.json', report)
        print(label, name, 'recorded', flush=True)
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('labels', nargs='+')
    args = parser.parse_args()
    existing = subprocess.run(['docker', 'inspect', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if existing.returncode == 0:
        raise SystemExit(f'Container {NAME} already exists; refusing to overwrite an active comparison.')
    args.output.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        directory = args.output / label
        directory.mkdir(exist_ok=True)
        launch_args = launch(label)
        before = mem_available()
        save(directory / 'launch.json', {'argv': launch_args, 'mem_available_before_bytes': before, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        started = time.monotonic()
        try:
            command(launch_args)
            while True:
                try:
                    get('/v1/models')
                    break
                except Exception:
                    state = subprocess.check_output(['docker', 'inspect', '-f', '{{.State.Running}}', NAME], text=True).strip()
                    if state != 'true' or time.monotonic() - started > 2400:
                        raise RuntimeError('Server failed to become ready')
                    time.sleep(5)
            ready = mem_available()
            startup_seconds = time.monotonic() - started
            print(label, 'READY', round(startup_seconds, 1), flush=True)
            report = benchmark(directory, label, quality_only=(label == 'bf16-control'))
            report['resources'] = {'startup_seconds': startup_seconds, 'mem_available_before_bytes': before, 'mem_available_ready_bytes': ready, 'mem_available_after_tests_bytes': mem_available()}
            save(directory / 'report.json', report)
        except Exception as error:
            save(directory / 'error.json', {'error': str(error)})
            print(label, 'ERROR', repr(error), flush=True)
        finally:
            with (directory / 'server.log').open('w') as log:
                subprocess.run(['docker', 'logs', NAME], stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(['docker', 'inspect', NAME], stdout=(directory / 'container.json').open('w'))
            subprocess.run(['docker', 'rm', '-f', NAME], check=False)

if __name__ == '__main__':
    main()
