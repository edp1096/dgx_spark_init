"""Keep weights resident; release per-request graph results and allocator slack."""
import ctypes
import gc
import logging
import os

import psutil
import torch
import nodes
import folder_paths
import execution
from aiohttp import web
from server import PromptServer
import comfy.model_management as memory

_clip_name = None
_clip = None
_WEIGHT_NODES = {'UNETLoader', 'VAELoader', 'CLIPLoader', 'LoraLoader', 'LoraLoaderModelOnly'}


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


def release_results(cache):
    if not cache.initialized:
        return
    keep = set()
    retained = {}
    for node_id in cache.cache_key_set.all_node_ids():
        if cache.dynprompt.get_node(node_id)['class_type'] in _WEIGHT_NODES:
            key = cache.cache_key_set.get_data_key(node_id)
            keep.add(key)
            if key in cache.cache:
                retained[node_id] = id(cache.cache[key])
    for key in list(cache.cache):
        if key not in keep:
            del cache.cache[key]
    for child in cache.subcaches.values():
        release_results(child)
    logging.info('SparkTalk retained weight cache: %s', retained)


_original_execute = execution.PromptExecutor.execute


def execute_and_release_workspace(self, *args, **kwargs):
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


execution.PromptExecutor.execute = execute_and_release_workspace
NODE_CLASS_MAPPINGS = {'SparkTalkTextEncode': SparkTalkTextEncode}


@PromptServer.instance.routes.get('/sparktalk/memory')
async def memory_state(request):
    ready = (_clip is not None and
             _clip.patcher.loaded_size() >= .9 * _clip.patcher.model_size())
    return web.json_response({'text_weights_cached': _clip is not None,
                              'text_weights_on_device': ready,
                              'workspace_gib': 2.5 if ready else 4.5})
