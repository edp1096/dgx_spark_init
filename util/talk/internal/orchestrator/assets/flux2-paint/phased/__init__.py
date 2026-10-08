"""Keep weights resident; release per-request graph results and allocator slack."""
import ctypes
import gc
import logging
import os
import asyncio
import threading
import weakref
import math

import psutil
import torch
import nodes
import folder_paths
import execution
from aiohttp import web
from server import PromptServer
import comfy.model_management as memory
import comfy.sd as sd
from comfy.weight_adapter.bypass import BypassInjectionManager

_clip_name = None
_clip = None
_core = None
_vae = None
_core_key = None
_vae_name = None
_execution_lock = threading.RLock()
_busy = threading.Event()
_executors = weakref.WeakSet()
_WEIGHT_NODES = {'UNETLoader', 'VAELoader', 'SparkTalkUNETLoader', 'SparkTalkVAELoader', 'CLIPLoader', 'LoraLoader', 'LoraLoaderModelOnly', 'SparkTalkBFSLoader'}
_LORA_NODES = {'LoraLoader', 'LoraLoaderModelOnly', 'SparkTalkBFSLoader'}


class SparkTalkBFSLoader(nodes.LoraLoaderModelOnly):
    def load_lora_model_only(self, model, lora_name, strength_model):
        if lora_name != 'bfs-head-v1-flux2-klein-4b.safetensors':
            raise RuntimeError('BFS loader accepts only the pinned Klein 4B V1 adapter')
        clone = super().load_lora_model_only(model, lora_name, strength_model)[0]
        manager = BypassInjectionManager()
        expected = 0
        for key, patches in clone.patches.items():
            if len(patches) != 1:
                raise RuntimeError('BFS must not be combined with another weight patch')
            strength, adapter, strength_base, offset, function = patches[0]
            if getattr(adapter, 'name', None) != 'lora' or strength_base != 1 or offset is not None or function is not None:
                raise RuntimeError('Unsupported BFS adapter patch; core unchanged')
            manager.add_adapter(key, adapter, strength=strength)
            expected += 1
        injections = manager.create_injections(clone.model)
        if expected != 80 or manager.get_hook_count() != expected:
            raise RuntimeError(f'BFS mapping incomplete: {manager.get_hook_count()}/{expected} layers; expected 80')
        clone.set_injections('sparktalk_bfs', injections)
        clone.patches.clear()
        logging.info('SparkTalk BFS V1: %d BF16 bypass layers; NVFP4 core weights unchanged', expected)
        return (clone,)


class SparkTalkUNETLoader(nodes.UNETLoader):
    def load_unet(self, unet_name, weight_dtype):
        global _core, _core_key
        key = (unet_name, weight_dtype)
        if _core is not None and _core_key != key:
            raise RuntimeError('Core selection changed; restart the image service explicitly')
        if _core is None:
            if weight_dtype != 'default':
                raise RuntimeError('Resident NVFP4 core requires its original weight dtype')
            path = folder_paths.get_full_path_or_raise('diffusion_models', unet_name)
            # AIMDO may evict the dynamic core's pages during VAE/text work even
            # when Comfy's model eviction list protects the patcher. Use the
            # upstream static loader for this small resident NVFP4 core only.
            _core = sd.load_diffusion_model(path, disable_dynamic=True)
            _core_key = key
        return (_core,)


class SparkTalkVAELoader(nodes.VAELoader):
    def load_vae(self, vae_name):
        global _vae, _vae_name
        if _vae is not None and _vae_name != vae_name:
            raise RuntimeError('VAE selection changed; restart the image service explicitly')
        if _vae is None:
            _vae = super().load_vae(vae_name)[0]
            _vae_name = vae_name
        return (_vae,)


class SparkTalkReferenceScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'max_pixels': ('INT', {'default': 1048576, 'min': 256, 'max': 1048576})}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'scale'
    CATEGORY = 'SparkTalk'

    def scale(self, image, max_pixels):
        height, width = image.shape[1:3]
        factor = min(1.0, math.sqrt(max_pixels / (width * height)))
        scaled_width = max(16, int(width * factor) // 16 * 16)
        scaled_height = max(16, int(height * factor) // 16 * 16)
        # Keep full portraits; multiple references share one pixel budget.
        return nodes.ImageScale().upscale(image, 'lanczos', scaled_width, scaled_height, 'disabled')


class SparkTalkWarmupConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',)}}
    RETURN_TYPES = ('CONDITIONING',)
    FUNCTION = 'conditioning'
    CATEGORY = 'SparkTalk'

    def conditioning(self, model):
        config = model.model.model_config.unet_config
        dimension = config['context_in_dim']
        metadata = {}
        vector_dimension = config.get('vec_in_dim')
        if isinstance(vector_dimension, int) and vector_dimension > 0:
            metadata['pooled_output'] = torch.zeros((1, vector_dimension), dtype=torch.float32)
        return ([[torch.zeros((1, 8, dimension), dtype=torch.float32), metadata]],)


# Preserve core weights and all LoRA views of that same model during Comfy's
# automatic memory eviction. A request that cannot fit must fail, not evict it.
_original_free_memory = memory.free_memory


def free_optional_memory(memory_required, device, keep_loaded=None, for_dynamic=False, pins_required=0, ram_required=0):
    keep = list(keep_loaded or [])
    if _core is not None:
        for loaded in memory.current_loaded_models:
            if loaded.model is not None and loaded.model.model is _core.model:
                if loaded not in keep:
                    keep.append(loaded)
    return _original_free_memory(memory_required, device, keep_loaded=keep,
                                 for_dynamic=for_dynamic, pins_required=pins_required, ram_required=ram_required)


memory.free_memory = free_optional_memory


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu')
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


class SparkTalkTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'text': ('STRING', {'multiline': True}),
                             'clip_name': (folder_paths.get_filename_list('text_encoders'),)}}
    RETURN_TYPES = ('CONDITIONING',)
    FUNCTION = 'encode'
    CATEGORY = 'SparkTalk'

    def encode(self, text, clip_name):
        global _clip, _clip_name
        if _clip is None or _clip_name != clip_name:
            # Only a model selection change replaces these weights.
            if _clip is not None:
                memory.unload_model_and_clones(_clip.patcher)
            _clip = nodes.CLIPLoader().load_clip(clip_name, type='flux2')[0]
            _clip_name = clip_name
            logging.info('SparkTalk text weights loaded: %s object=%s', clip_name, id(_clip))
        else:
            logging.info('SparkTalk text weights reused: object=%s', id(_clip))
        result = cpu_tree(nodes.CLIPTextEncode().encode(_clip, text))
        # Keep the diffusion core/VAE resident. The optional text model yields
        # only when the measured headroom cannot cover diffusion workspace.
        if psutil.virtual_memory().available < 6.5 * 1024**3:
            memory.unload_model_and_clones(_clip.patcher)
            memory.soft_empty_cache(force=True)
            _clip = None
            _clip_name = None
            gc.collect()
            memory.cleanup_models()
            memory.soft_empty_cache(force=True)
            ctypes.CDLL(None).malloc_trim(0)
            logging.info('SparkTalk optional text weights released under pressure; diffusion core retained')
        return result


def release_results(cache, keep_lora=True):
    if not cache.initialized:
        return
    keep = set()
    retained = {}
    for node_id in cache.cache_key_set.all_node_ids():
        kind = cache.dynprompt.get_node(node_id)['class_type']
        if kind in _WEIGHT_NODES and (keep_lora or kind not in _LORA_NODES):
            key = cache.cache_key_set.get_data_key(node_id)
            keep.add(key)
            if key in cache.cache:
                retained[node_id] = id(cache.cache[key])
    for key in list(cache.cache):
        if key not in keep:
            del cache.cache[key]
    for child in cache.subcaches.values():
        release_results(child, keep_lora)
    logging.info('SparkTalk retained weight cache: %s', retained)


_original_execute = execution.PromptExecutor.execute


def _execute_and_release_workspace(self, *args, **kwargs):
    global _clip, _clip_name
    try:
        return _original_execute(self, *args, **kwargs)
    finally:
        # The worker is serialized, and execute() has returned: no live graph
        # can still use these images, conditioning tensors or sampler objects.
        release_results(self.caches.outputs)
        gc.collect()
        memory.soft_empty_cache(force=True)
        ctypes.CDLL(None).malloc_trim(0)
        # Retain the diffusion core and VAE. Only evict the optional text model
        # if workspace cleanup still leaves less than the configured reserve.
        reserve = float(os.getenv('SPARKTALK_FLUX_IDLE_RESERVE_GIB', '1.5')) * 1024**3
        if _clip is not None and psutil.virtual_memory().available < reserve:
            memory.unload_model_and_clones(_clip.patcher)
            gc.collect()
            memory.soft_empty_cache(force=True)
            ctypes.CDLL(None).malloc_trim(0)
            logging.info('SparkTalk text CUDA allocation yielded under pressure; weights object retained')
        logging.info('SparkTalk workspace released; live weights preserved')


def execute_and_release_workspace(self, *args, **kwargs):
    with _execution_lock:
        _executors.add(self)
        _busy.set()
        try:
            return _execute_and_release_workspace(self, *args, **kwargs)
        finally:
            _busy.clear()


execution.PromptExecutor.execute = execute_and_release_workspace
NODE_CLASS_MAPPINGS = {'SparkTalkTextEncode': SparkTalkTextEncode,
                      'SparkTalkUNETLoader': SparkTalkUNETLoader,
                      'SparkTalkVAELoader': SparkTalkVAELoader,
                      'SparkTalkBFSLoader': SparkTalkBFSLoader,
                      'SparkTalkReferenceScale': SparkTalkReferenceScale,
                      'SparkTalkWarmupConditioning': SparkTalkWarmupConditioning}


def memory_snapshot():
    clip = _clip
    core = _core
    ready = clip is not None and clip.patcher.loaded_size() >= .9 * clip.patcher.model_size()
    core_loaded = 0
    for loaded in list(memory.current_loaded_models):
        patcher = loaded.model
        if patcher is not None and core is not None and patcher.model is core.model:
            core_loaded = max(core_loaded, patcher.loaded_size())
    core_ready = core is not None and core_loaded > 0 and core_loaded >= .9 * core.model_size()
    return {'status': 'ok', 'busy': _busy.is_set(),
            'memory_schema': 1, 'workspace_kind': 'additional',
            'cuda_allocated_gib': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            'cuda_reserved_gib': torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0,
            'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            'core_ready': core_ready, 'core_object': id(_core.model) if _core is not None else None,
            'core_loaded_gib': core_loaded / 1024**3,
            'text_weights_cached': _clip is not None, 'text_weights_on_device': ready,
            # 1024px cold-text generation grew host unified occupancy by
            # 6.17 GiB after core preparation (2026-10-01). Text residency
            # alone recovered ~3.5 GiB. Keep 4 GiB for its reload/staging,
            # separate from the 2.5 GiB generation workspace.
            'text_reload_gib': 0.0 if ready else 4.0,
            'generation_workspace_gib': 2.5,
            'workspace_gib': 2.5 + (0.0 if ready else 4.0)}


@PromptServer.instance.routes.get('/sparktalk/memory')
async def memory_state(request):
    return web.json_response(memory_snapshot())


def control_runtime(action, payload):
    global _clip, _clip_name
    if not _execution_lock.acquire(blocking=False):
        raise RuntimeError('Image work is active; weights cannot be changed')
    _busy.set()
    try:
        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if action == 'prepare':
                core = SparkTalkUNETLoader().load_unet(payload['unet_name'], 'default')[0]
                vae = SparkTalkVAELoader().load_vae(payload['vae_name'])[0]
                # Dynamic load registers vbars but fills them only during
                # forward execution. The API follows this with a tiny graph
                # if the measured loaded_size is still below the ready level.
                memory.load_models_gpu([core], force_full_load=True)
                memory.load_models_gpu([vae.patcher])
            elif action == 'reclaim':
                if _clip is not None:
                    memory.unload_model_and_clones(_clip.patcher)
                    _clip = None
                    _clip_name = None
                # Dynamic unpatch_model() frees the shared core's vbars too.
                # Release adapter/prepared tensors only. The next managed load
                # switches to the canonical core (or a fresh LoRA clone) using
                # Comfy's normal dirty/patch UUID path before any forward pass.
                if _core is not None:
                    for loaded in list(memory.current_loaded_models):
                        patcher = loaded.model
                        if patcher is not None and patcher.model is _core.model and patcher is not _core:
                            if not patcher.is_dynamic():
                                # Static LoRA clones share the resident model.
                                # Restore only adapter-modified weights, without
                                # moving the complete model back to CPU.
                                patcher.unpatch_model(device_to=patcher.load_device)
                                patcher.remove_injections('sparktalk_bfs')
                                # Upstream unpatch resets this accounting field
                                # even after .to(cuda) restored every weight.
                                # Keep residency reporting aligned with the
                                # retained full static core, avoiding a false
                                # not-ready result and another preload.
                                patcher.model.model_loaded_weight_memory = _core.model_size()
                                continue
                            for module in patcher.model.modules():
                                for attr in ('weight_function', 'bias_function'):
                                    functions = getattr(module, attr, ())
                                    if not isinstance(functions, (list, tuple)):
                                        functions = (functions,)
                                    for function in functions:
                                        if hasattr(function, 'clear_prepared'):
                                            function.clear_prepared()
                            patcher.patches.clear()
                for executor in list(_executors):
                    release_results(executor.caches.outputs, keep_lora=False)
                    release_results(executor.caches.objects, keep_lora=False)
            else:
                raise ValueError('Unknown runtime action')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            memory.cleanup_models()
            memory.soft_empty_cache(force=True)
            ctypes.CDLL(None).malloc_trim(0)
            state = memory_snapshot()
            state['busy'] = False
            logging.info('SparkTalk %s: core_object=%s core_loaded_gib=%.3f text_cached=%s',
                         action, state['core_object'], state['core_loaded_gib'], state['text_weights_cached'])
            return state
    finally:
        _busy.clear()
        _execution_lock.release()


@PromptServer.instance.routes.post('/sparktalk/prepare')
async def prepare_runtime(request):
    payload = await request.json()
    if not isinstance(payload, dict) or not all(isinstance(payload.get(k), str) for k in ('unet_name', 'vae_name')):
        raise web.HTTPBadRequest(text='Model filenames are required')
    try:
        return web.json_response(await asyncio.to_thread(control_runtime, 'prepare', payload))
    except RuntimeError as error:
        raise web.HTTPConflict(text=str(error))


@PromptServer.instance.routes.post('/sparktalk/reclaim')
async def reclaim_runtime(request):
    try:
        return web.json_response(await asyncio.to_thread(control_runtime, 'reclaim', {}))
    except RuntimeError as error:
        raise web.HTTPConflict(text=str(error))
