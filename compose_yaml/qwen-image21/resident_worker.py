"""Qualified Qwen Image DiT stays on GPU; request-owned TE/VAE are discarded."""
import ctypes
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager

ROOT = Path(os.getenv('JOB_DIR', '/job'))
SETTINGS = json.loads((ROOT / 'settings.json').read_text())
cuda = ctypes.CDLL('libcuda.so.1')
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
assert cuda.cuInit(0) == 0
primary = ctypes.c_void_p()
assert cuda.cuDevicePrimaryCtxRetain(ctypes.byref(primary), 0) == 0
assert cuda.cuCtxSetCurrent(primary) == 0
sys.path.insert(0, '/opt/ComfyUI')
sys.argv = ['resident-worker', '--disable-dynamic-vram', '--disable-pinned-memory',
            '--use-pytorch-cross-attention', '--reserve-vram', '2', '--preview-method', 'none']
import comfy.options
comfy.options.enable_args_parsing()
AUX_DYNAMIC = SETTINGS.get('aux_loading', 'static') == 'dynamic'
if AUX_DYNAMIC:
    import comfy_aimdo.control
    comfy_aimdo.control.init(simple_vram_headroom=2 * 1024**3, nvml_pressure=True)
import torch
import comfy.sd
import comfy.samplers
import comfy.model_management as mm
import folder_paths
import nodes
from comfy_extras.nodes_qwen import TextEncodeQwenImage21
from PIL import Image

if AUX_DYNAMIC:
    if not comfy_aimdo.control.init_devices((d.index, 0) for d in mm.get_all_torch_devices()):
        raise RuntimeError('Dynamic auxiliary allocator initialization failed')
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True

logging.basicConfig(level=logging.INFO)
for kind, directories in SETTINGS['model_paths'].items():
    for directory in directories:
        folder_paths.add_model_folder_path(kind, directory)
folder_paths.set_input_directory(os.getenv('IMAGE_INPUT_DIR','/tmp/qwen-image21/input'))
for directory in ('requests', 'results', 'output', 'cancel'):
    (ROOT / directory).mkdir(exist_ok=True)

DITS = {}
CONDITIONING = OrderedDict()
EVENTS = (ROOT / 'events.jsonl').open('a', buffering=1)
CASE = 'startup'
TIMINGS = {}


def emit(event, **data):
    if CASE != 'startup' and (ROOT/'cancel'/CASE).exists() and event != 'idle':
        raise JobCanceled(CASE)
    torch.cuda.synchronize()
    memory = {k: int(v.split()[0]) * 1024 for k, v in
              (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    item = dict(time=time.time(), case=CASE, event=event,
                available_GiB=memory['MemAvailable'] / 2**30, free_GiB=memory['MemFree'] / 2**30,
                allocated_GiB=torch.cuda.memory_allocated() / 2**30,
                reserved_GiB=torch.cuda.memory_reserved() / 2**30,
                loaded_models=[type(m.model.model).__name__ for m in mm.current_loaded_models if m.model is not None],
                resident={k: v.loaded_size() / 2**30 for k, v in DITS.items()}, **data)
    EVENTS.write(json.dumps(item) + '\n')
    (ROOT / 'phase').write_text(CASE + ':' + event)
    state = ROOT/'worker-state.tmp'
    state.write_text(json.dumps(item))
    state.replace(ROOT/'worker-state.json')
    print('STAGE ' + json.dumps(item), flush=True)
    return item


comfy.utils.set_progress_bar_global_hook(lambda value, total, preview=None, **kwargs: emit('progress', step=value, total=total))


# Respect normal Comfy admission but never select the persistent DiT as an
# eviction victim. Fail the request if its working set cannot fit.
original_free = mm.free_memory
def free_memory(memory_required, device, keep_loaded=None, **kwargs):
    keep = list(keep_loaded or [])
    for loaded in mm.current_loaded_models:
        if any(loaded.model is dit for dit in DITS.values()) and loaded not in keep:
            keep.append(loaded)
    return original_free(memory_required, device, keep, **kwargs)
mm.free_memory = free_memory
original_load = mm.load_models_gpu
def load_models_gpu(models, *args, **kwargs):
    kwargs['force_full_load'] = True
    return original_load(models, *args, **kwargs)
mm.load_models_gpu = load_models_gpu


def cleanup():
    # Auxiliaries have no graph cache owner. Drop their GPU storage directly;
    # don't create a redundant CPU copy just to unload a discarded model.
    gc.collect()
    mm.cleanup_models()
    mm.soft_empty_cache(force=True)
    extra = [type(loaded.model.model).__name__ for loaded in mm.current_loaded_models if loaded.model is not None and not any(loaded.model is dit for dit in DITS.values())]
    if extra:
        raise RuntimeError('Auxiliary models retained after stage: ' + repr(extra))
    for name, model in DITS.items():
        if model.loaded_size() < model.model_size() * .99:
            raise RuntimeError(f'{name} DiT unexpectedly offloaded')


@contextmanager
def stage(name):
    emit(name + '_start')
    start = time.monotonic()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        TIMINGS[name] = time.monotonic() - start
        emit(name + '_end', seconds=TIMINGS[name])


class JobCanceled(Exception):
    pass


def encode(prompt, reference_files):
    clip = nodes.CLIPLoader().load_clip('qwen3vl_8b_w4a8.safetensors', 'qwen_image')[0]
    images = {f'image_{i}': nodes.LoadImage().load_image(name)[0]
              for i, name in enumerate(reference_files, 1)}
    vae = (nodes.VAELoader().load_vae('qwen_image_2.1_vae_bf16.safetensors')[0]
           if images else None)
    result = TextEncodeQwenImage21.execute(clip, prompt, '', vae=vae,
                                         resolution=1024, images=images)
    return result[0], result[1], result[2]


def conditioning(request):
    # Reference inputs are request-owned and may change despite an equal prompt.
    key = request['prompt'] if not request.get('reference_files') else None
    if key is not None and key in CONDITIONING:
        CONDITIONING.move_to_end(key)
        emit('conditioning_cache_hit')
        return CONDITIONING[key]
    value = encode(request['prompt'], request.get('reference_files', []))
    if key is not None:
        CONDITIONING[key] = value
        while len(CONDITIONING) > 2:
            CONDITIONING.popitem(last=False)
    return value


def sample(request, cond):
    if request.get('reference_files') and request['operation'] != 'reference_generate':
        latent = cond[2]
    else:
        latent = nodes.EmptyLatentImage().generate(request['width'], request['height'], 1)[0]
    return nodes.KSampler().sample(DITS['qwim'], request['seed'], 40, 1.0,
                                  'euler', 'simple', cond[0], cond[1], latent, denoise=1.0)[0]


def decode_image(latent):
    vae = nodes.VAELoader().load_vae('qwen_image_2.1_vae_bf16.safetensors')[0]
    return nodes.VAEDecode().decode(vae, latent)[0]


@torch.inference_mode()
def generate(request):
    global CASE, TIMINGS
    CASE = request['case']
    if not CASE or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in CASE):
        raise ValueError('invalid request ID')
    TIMINGS = {}
    started = time.monotonic()
    with stage('encode'):
        cond = conditioning(request)
    with stage('release_encoder'):
        cleanup()
    with stage('sample'):
        latent = sample(request, cond)
    with stage('release_sample_workspace'):
        cleanup()
    with stage('decode_image'):
        frames = decode_image(latent)
    with stage('release_vae'):
        cleanup()
    with stage('save'):
        path = ROOT/'output'/(CASE+'.png')
        Image.fromarray((frames[0].cpu().numpy().clip(0, 1)*255).astype('uint8')).save(path)
    return dict(request=request, seconds=time.monotonic()-started,
                stages=dict(TIMINGS), file=str(path), pid=os.getpid(), status='success')


with torch.inference_mode():
    for kind in ('qwim',):
        with stage('load_' + kind + '_dit'):
            model = comfy.sd.load_diffusion_model(SETTINGS['dits'][kind], model_options={}, disable_dynamic=True)
            if kind == 'qwim':
                model.model_options['transformer_options']['qwen_image21_cache'] = {'device': 'auto', 'dtype': 'int8'}
            mm.load_models_gpu([model])
            DITS[kind] = model
            del model
            cleanup()
(ROOT / 'ready.json').write_text(json.dumps(emit('ready'), indent=2))
while not (ROOT / 'stop').exists():
    requests = sorted((ROOT / 'requests').glob('*.json'))
    if not requests:
        time.sleep(.1)
        continue
    path = requests[0]
    request = json.loads(path.read_text())
    path.rename(path.with_suffix('.running'))
    try:
        result = generate(request)
        cleanup()
        result['idle'] = emit('idle')
    except Exception:
        canceled = (ROOT/'cancel'/request['case']).exists()
        result = dict(status='canceled' if canceled else 'error', error=traceback.format_exc(), request=request)
        print(result['error'], flush=True)
    if result['status'] != 'success':
        cleanup()
        emit('idle')
    destination = ROOT / 'results' / path.name
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(destination)
    (ROOT/'cancel'/request['case']).unlink(missing_ok=True)
