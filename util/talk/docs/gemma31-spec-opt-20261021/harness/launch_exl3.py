"""Use pinned cached weights and a config-only view for vision exclusions."""
import json
import os
from pathlib import Path
import sys

TARGET = 'lyf/Huihui-gemma-4-31B-it-abliterated-v2-NVFP4'
TARGET_REVISION = '985633e8001d0bfd9b621738234394744b992907'
ASSISTANT = 'edp1096/gemma-4-31B-it-assistant-exl3'
ASSISTANT_REVISION = '359650d0d9386fbe42b820c74eefa8075017ebc6'


def prepare(cache, runtime):
    def snapshot(repo, revision):
        return cache / 'hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
    target = Path("/job/models/target")
    kind = os.environ.get('DRAFT_KIND', 'assistant')
    assistant = Path('/job/models/' + kind) if kind in ('dflash', 'original') else Path('/job/models/b' + os.environ['ASSISTANT_BODY_BITS'])
    required = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'processor_config.json']
    if any(not (target / name).is_file() for name in required):
        raise RuntimeError('Target cache incomplete')
    if kind == 'dflash':
        if any(not (assistant / name).is_file() for name in ['model.safetensors', 'config.json']):
            raise RuntimeError('DFlash cache incomplete')
        if json.loads((assistant / 'config.json').read_text())['architectures'] != ['DFlashDraftModel']:
            raise RuntimeError('Wrong DFlash architecture')
    elif kind == 'original':
        if any(not (assistant / name).is_file() for name in ['model.safetensors', 'config.json']):
            raise RuntimeError('Google assistant cache incomplete')
        if json.loads((assistant / 'config.json').read_text())['architectures'] != ['Gemma4AssistantForCausalLM']:
            raise RuntimeError('Wrong Google assistant architecture')
    else:
        index = json.loads((assistant / 'model.safetensors.index.json').read_text())
        if any(not (assistant / name).is_file() for name in set(index['weight_map'].values())):
            raise RuntimeError('Assistant cache incomplete')
        quant = json.loads((assistant / 'quantization_config.json').read_text())
        if quant['quant_method'] != 'exl3' or quant['bits'] != int(os.environ['ASSISTANT_BODY_BITS']) or quant['head_bits'] != 6:
            raise RuntimeError('Assistant quantization mismatch')
    config = json.loads((target / 'config.json').read_text())
    ignore = config['quantization_config']['ignore']
    if 're:.*vision.*' not in ignore:
        ignore.append('re:.*vision.*')
    view = runtime / 'model-view'
    view.mkdir(parents=True, exist_ok=True)
    for path in target.iterdir():
        if path.name == 'config.json' or not path.is_file():
            continue
        dest = view / path.name
        if dest.is_symlink():
            if dest.readlink() == path:
                continue
            dest.unlink()
        elif dest.exists():
            raise RuntimeError('Unexpected regular file in model view: ' + dest.name)
        dest.symlink_to(path)
    temporary = view / 'config.json.tmp'
    temporary.write_text(json.dumps(config, indent=2) + '\n')
    temporary.replace(view / 'config.json')
    return view, assistant


if __name__ == '__main__':
    cache = Path(os.environ.get('HF_HOME', '/root/.cache/huggingface'))
    view, assistant = prepare(cache, Path('/root/.cache/sglang'))
    args = sys.argv[1:]
    replacements = {'--model-path': str(view), '--speculative-draft-model-path': str(assistant)}
    output = []
    i = 0
    while i < len(args):
        key = args[i]
        if key in ('--revision', '--speculative-draft-model-revision'):
            i += 2
            continue
        if key in replacements:
            output.extend([key, replacements[key]])
            i += 2
            continue
        output.append(key)
        i += 1
    print('GEMMA31_OPT_ARGS', json.dumps(output), flush=True)
    os.execv(sys.executable, [sys.executable, '-m', 'sglang.launch_server', *output])
