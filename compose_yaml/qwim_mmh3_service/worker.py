"""Serial worker: resident QWI/H3 DiTs, disposable auxiliary models.

Run in the pinned sparktalk-mmh3-swap-test image with /job/settings.json.
The host owns admission, memory monitoring, and restoration of the normal API.
Requests are atomic JSON files in /job/requests; results go to /job/results.
"""
import ctypes
import gc
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager

ROOT = Path('/job')
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
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
from comfy_extras.nodes_qwen import TextEncodeQwenImage21
from comfy_extras.nodes_custom_sampler import BasicScheduler, BasicGuider, RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_audio import VAEDecodeAudio
from comfy_extras.nodes_video import CreateVideo
from comfy_api.latest import Types
from PIL import Image

if AUX_DYNAMIC:
    if not comfy_aimdo.control.init_devices((d.index, 0) for d in mm.get_all_torch_devices()):
        raise RuntimeError('Dynamic auxiliary allocator initialization failed')
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, '/opt/ComfyUI/custom_nodes')
clipproj = importlib.import_module('ComfyUI-ClipProj')
for kind, directories in SETTINGS['model_paths'].items():
    for directory in directories:
        folder_paths.add_model_folder_path(kind, directory)
for directory in ('requests', 'results', 'output'):
    (ROOT / directory).mkdir(exist_ok=True)

DITS = {}
CONDITIONING = OrderedDict()
EVENTS = (ROOT / 'events.jsonl').open('a', buffering=1)
CASE = 'startup'
TIMINGS = {}
TRACKER = None


def emit(event, **data):
    torch.cuda.synchronize()
    memory = {k: int(v.split()[0]) * 1024 for k, v in
              (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    item = dict(time=time.time(), case=CASE, event=event,
                available_GiB=memory['MemAvailable'] / 2**30,
                allocated_GiB=torch.cuda.memory_allocated() / 2**30,
                reserved_GiB=torch.cuda.memory_reserved() / 2**30,
                loaded_models=[type(m.model.model).__name__ for m in mm.current_loaded_models if m.model is not None],
                resident={k: v.loaded_size() / 2**30 for k, v in DITS.items()}, **data)
    EVENTS.write(json.dumps(item) + '\n')
    if TRACKER is not None:
        TRACKER.record(event, **data)
    (ROOT / 'phase').write_text(CASE + ':' + event)
    print('STAGE ' + json.dumps(item), flush=True)
    return item


comfy.utils.set_progress_bar_global_hook(lambda value, total, preview=None, **kwargs: emit('progress', step=value, total=total))


# Respect normal Comfy admission but never select either persistent DiT as an
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


def encode(kind, prompt):
    if kind == 'h3':
        clip = nodes.CLIPLoader().load_clip('qwen3vl_4b_fp8_scaled.safetensors', 'krea2')[0]
        clip = clipproj.NODE_CLASS_MAPPINGS['ClipProjApply']().apply(
            clip, 'mmh3-4b-ClipProj-v3.1.safetensors')[0]
        return MiniMaxH3ImageToVideo.execute(clip, None, prompt, 864, 480, 124)[0]
    clip = nodes.CLIPLoader().load_clip('qwen3vl_8b_w4a8.safetensors', 'qwen_image')[0]
    encoded = TextEncodeQwenImage21.execute(clip, prompt, '', resolution=1024)
    return encoded[0], encoded[1]


def conditioning(kind, prompt):
    key = (kind, prompt)
    if key in CONDITIONING:
        CONDITIONING.move_to_end(key)
        emit('conditioning_cache_hit')
        return CONDITIONING[key]
    value = encode(kind, prompt)
    CONDITIONING[key] = value
    while len(CONDITIONING) > 2:
        CONDITIONING.popitem(last=False)
    return value


def sample(kind, cond, seed):
    model = DITS[kind]
    if kind == 'h3':
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent
        latent, _ = _empty_av_latent(864, 480, 124)
        guider = BasicGuider.execute(model, cond)[0]
        sigmas = BasicScheduler.execute(model, 'simple', 20, 1.0)[0]
        return SamplerCustomAdvanced.execute(RandomNoise.execute(seed)[0], guider,
                    comfy.samplers.sampler_object('res_multistep'), sigmas, latent)[0]
    latent = nodes.EmptyLatentImage().generate(1024, 1024, 1)[0]
    return nodes.KSampler().sample(model, seed, 40, 1.0, 'euler', 'simple',
                                  cond[0], cond[1], latent, denoise=1.0)[0]


def decode_video(kind, latent):
    filename = ('minimax_h3_video_vae_int8_convrot.safetensors' if kind == 'h3'
                else 'qwen_image_2.1_vae_bf16.safetensors')
    vae = nodes.VAELoader().load_vae(filename)[0]
    return nodes.VAEDecode().decode(vae, latent)[0]


def decode_audio(latent):
    vae = nodes.VAELoader().load_vae('minimax_h3_audio_vae_fp32.safetensors')[0]
    return VAEDecodeAudio.execute(vae, latent)[0]


@torch.inference_mode()
def generate(request):
    global CASE, TIMINGS, TRACKER
    CASE = request['case']
    if not CASE or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in CASE):
        raise ValueError('case must be a lowercase filename stem')
    TIMINGS = {}
    kind = request['kind']
    if kind not in DITS:
        raise ValueError('kind must be h3 or qwim')
    from generation_progress import Tracker
    TRACKER = Tracker(ROOT, os.getenv('STATE_DIR', '/job/state'), request.get('progress_id', CASE), kind)
    start = time.monotonic()
    with stage('encode'):
        cond = conditioning(kind, request['prompt'])
    with stage('release_encoder'):
        cleanup()
    with stage('sample'):
        latent = sample(kind, cond, request['seed'])
    with stage('release_sample_workspace'):
        cleanup()
    with stage('decode_video'):
        frames = decode_video(kind, latent)
    with stage('release_video_vae'):
        cleanup()
    audio = None
    if kind == 'h3':
        with stage('decode_audio'):
            audio = decode_audio(latent)
        with stage('release_audio_vae'):
            cleanup()
    with stage('save'):
        path = ROOT / 'output' / (CASE + ('.mp4' if kind == 'h3' else '.png'))
        if kind == 'h3':
            video = CreateVideo.execute(frames, 24.0, audio)[0]
            video.save_to(str(path), format=Types.VideoContainer('mp4'), codec=Types.VideoCodec('h264'))
        else:
            Image.fromarray((frames[0].cpu().numpy().clip(0, 1) * 255).astype('uint8')).save(path)
    total = time.monotonic() - start
    return dict(request=request, seconds=total, stages=dict(TIMINGS), file=str(path),
                pid=os.getpid(), status='success')


with torch.inference_mode():
    for kind in ('h3', 'qwim'):
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
        try:
            TRACKER.remember(result['stages'])
        except OSError:
            logging.warning('Could not save ETA timing history', exc_info=True)
        result['idle'] = emit('idle')
        TRACKER = None
    except Exception:
        result = dict(status='error', error=traceback.format_exc(), request=request)
        if TRACKER is not None:
            TRACKER.record('failed')
        print(result['error'], flush=True)
    destination = ROOT / 'results' / path.name
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(destination)
    if result['status'] == 'error':
        raise SystemExit(1)
