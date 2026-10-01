"""Lifecycle tests without CUDA, model downloads, Docker, or listening sockets."""
import contextlib
import ast
import asyncio
import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
import uuid
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'internal/orchestrator/assets/flux2-paint/phased/__init__.py'


class Patcher:
    def __init__(self, model=None):
        self.model = model if model is not None else types.SimpleNamespace(modules=lambda: [])
        self.patches = {'adapter': object()}
        self.unpatched = False
        self.load_device = 'cuda'

    def model_size(self): return 1024
    def loaded_size(self): return 1024
    def unpatch_model(self, **kwargs): self.unpatched = True; self.unpatch_device = kwargs.get('device_to')
    def remove_injections(self, key): pass
    def is_dynamic(self): return True


class Cache:
    def __init__(self, kinds):
        self.initialized = True
        self.cache = {key: object() for key in kinds}
        self.subcaches = {}
        self.cache_key_set = types.SimpleNamespace(all_node_ids=lambda: list(kinds), get_data_key=lambda key: key)
        self.dynprompt = types.SimpleNamespace(get_node=lambda key: {'class_type': kinds[key]})


class ResidentTests(unittest.TestCase):
    def setUp(self):
        self.loads = []
        self.evicted = []
        self.kept = []
        outer = self
        class LoraLoaderModelOnly: pass
        class UNETLoader:
            def load_unet(self, name, dtype):
                outer.loads.append(('core', name)); return (Patcher(),)
        class VAELoader:
            def load_vae(self, name):
                outer.loads.append(('vae', name)); return (types.SimpleNamespace(patcher=Patcher()),)
        class Executor:
            def execute(self, *args, **kwargs): return 'completed'
        class ImageScale:
            def upscale(self, image, method, width, height, crop):
                return (types.SimpleNamespace(shape=(1, height, width, 3), crop=crop),)
        class Routes:
            def get(self, route): return lambda function: function
            def post(self, route): return lambda function: function
        def free_memory(required, device, keep_loaded=None, **kwargs):
            outer.kept = keep_loaded
            return []
        memory = types.ModuleType('comfy.model_management')
        memory.current_loaded_models = []
        memory.free_memory = free_memory
        memory.unload_model_and_clones = self.evicted.append
        memory.soft_empty_cache = lambda **kwargs: None
        memory.cleanup_models = lambda: None
        def load_models(models, **kwargs):
            memory.current_loaded_models.extend(types.SimpleNamespace(model=p) for p in models if not any(x.model is p for x in memory.current_loaded_models))
        memory.load_models_gpu = load_models
        modules = {
            'comfy': types.ModuleType('comfy'), 'comfy.model_management': memory,
            'comfy.weight_adapter': types.ModuleType('comfy.weight_adapter'),
            'comfy.weight_adapter.bypass': types.SimpleNamespace(BypassInjectionManager=type('BypassInjectionManager', (), {})),
            'comfy.sd': types.SimpleNamespace(load_diffusion_model=lambda path, **kwargs: UNETLoader().load_unet(path, 'default')[0]),
            'torch': types.SimpleNamespace(Tensor=type('Tensor', (), {}), inference_mode=contextlib.nullcontext,
                                          cuda=types.SimpleNamespace(is_available=lambda: False)),
            'psutil': types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=20*1024**3)),
            'nodes': types.SimpleNamespace(LoraLoaderModelOnly=LoraLoaderModelOnly, UNETLoader=UNETLoader, VAELoader=VAELoader, ImageScale=ImageScale),
            'folder_paths': types.SimpleNamespace(get_full_path_or_raise=lambda category, name: name),
            'execution': types.SimpleNamespace(PromptExecutor=Executor),
            'aiohttp': types.SimpleNamespace(web=types.SimpleNamespace()),
            'server': types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=Routes()))),
            'ctypes': types.SimpleNamespace(CDLL=lambda unused: types.SimpleNamespace(malloc_trim=lambda unused: None)),
        }
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location('resident_test_module', SOURCE)
            self.runtime = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.runtime)
        self.memory = memory
        self.Executor = Executor

    def prepare(self):
        return self.runtime.control_runtime('prepare', {'unet_name': 'core.safetensors', 'vae_name': 'vae.safetensors'})

    def test_workspace_separates_text_reload_from_generation(self):
        cold = self.runtime.memory_snapshot()
        self.assertEqual(cold['memory_schema'], 1)
        self.assertEqual(cold['workspace_kind'], 'additional')
        self.assertEqual(cold['workspace_gib'], 6.5)
        self.assertEqual(cold['text_reload_gib'], 4)
        self.runtime._clip = types.SimpleNamespace(patcher=Patcher())
        warm = self.runtime.memory_snapshot()
        self.assertEqual(warm['workspace_gib'], 2.5)
        self.assertEqual(warm['text_reload_gib'], 0)

    def test_bfs_bypass_maps_every_layer_and_does_not_merge_core_weights(self):
        patches = {f'layer{i}.weight': [(1.0, types.SimpleNamespace(name='lora'), 1.0, None, None)] for i in range(80)}
        clone = types.SimpleNamespace(patches=patches.copy(), model=object())
        clone.set_injections = lambda key, injection: setattr(clone, 'injection', (key, injection))
        base = self.runtime.SparkTalkBFSLoader.__bases__[0]
        base.load_lora_model_only = lambda *args: (clone,)
        class Manager:
            def __init__(self): self.keys = []
            def add_adapter(self, key, adapter, strength): self.keys.append(key)
            def create_injections(self, model): return ['injection']
            def get_hook_count(self): return len(self.keys)
        self.runtime.BypassInjectionManager = Manager
        loaded = self.runtime.SparkTalkBFSLoader().load_lora_model_only(object(), 'bfs-head-v1-flux2-klein-4b.safetensors', 1)[0]
        self.assertIs(loaded, clone)
        self.assertEqual(loaded.patches, {})
        self.assertEqual(loaded.injection, ('sparktalk_bfs', ['injection']))
        self.assertEqual(len(patches), 80)
        clone.patches = dict(list(patches.items())[:79])
        with self.assertRaisesRegex(RuntimeError, 'mapping incomplete'):
            self.runtime.SparkTalkBFSLoader().load_lora_model_only(object(), 'bfs-head-v1-flux2-klein-4b.safetensors', 1)

    def test_reference_portrait_keeps_full_frame_with_shared_pixel_budget(self):
        source = types.SimpleNamespace(shape=(1, 703, 423, 3))
        scaled = self.runtime.SparkTalkReferenceScale().scale(source, 262144)[0]
        self.assertEqual(scaled.crop, 'disabled')
        self.assertLessEqual(scaled.shape[1] * scaled.shape[2], 262144)
        self.assertLess(abs(scaled.shape[2] / scaled.shape[1] - 423 / 703), .04)

    def test_preload_and_graph_use_same_core_and_vae(self):
        first = self.prepare()
        core = self.runtime.SparkTalkUNETLoader().load_unet('core.safetensors', 'default')[0]
        vae = self.runtime.SparkTalkVAELoader().load_vae('vae.safetensors')[0]
        second = self.prepare()
        self.assertTrue(first['core_ready'])
        self.assertEqual(first['core_object'], second['core_object'])
        self.assertIs(core, self.runtime._core)
        self.assertIs(vae, self.runtime._vae)
        self.assertEqual(self.loads, [('core', 'core.safetensors'), ('vae', 'vae.safetensors')])

    def test_reclaim_drops_lora_and_results_without_unpatching_shared_core(self):
        self.prepare()
        core = self.runtime._core
        clone = Patcher(core.model)
        prepared = types.SimpleNamespace(value=object())
        prepared.clear_prepared = lambda: setattr(prepared, 'value', None)
        clone.model.modules = lambda: [types.SimpleNamespace(weight_function=[prepared], bias_function=[])]
        self.memory.current_loaded_models.append(types.SimpleNamespace(model=clone))
        text = Patcher()
        self.runtime._clip = types.SimpleNamespace(patcher=text)
        executor = self.Executor()
        kinds = {'core': 'SparkTalkUNETLoader', 'vae': 'SparkTalkVAELoader', 'lora': 'LoraLoaderModelOnly', 'latent': 'SamplerCustomAdvanced'}
        executor.caches = types.SimpleNamespace(outputs=Cache(kinds), objects=Cache(kinds))
        self.runtime._executors.add(executor)
        state = self.runtime.control_runtime('reclaim', {})
        self.assertEqual(self.evicted, [text])
        self.assertIs(core, self.runtime._core)
        self.assertFalse(clone.unpatched)
        self.assertFalse(clone.patches)
        self.assertIsNone(prepared.value)
        self.assertFalse(state['text_weights_cached'])
        self.assertEqual(set(executor.caches.outputs.cache), {'core', 'vae'})
        self.assertEqual(set(executor.caches.objects.cache), {'core', 'vae'})

    def test_comfy_eviction_protects_core_and_its_lora_clone(self):
        self.prepare()
        clone = types.SimpleNamespace(model=Patcher(self.runtime._core.model))
        self.memory.current_loaded_models.append(clone)
        self.runtime.free_optional_memory(100, None)
        self.assertEqual(len(self.kept), 2)
        self.assertIn(clone, self.kept)
        self.assertTrue(all(x.model.model is self.runtime._core.model for x in self.kept))

    def test_active_execution_cannot_be_reclaimed(self):
        self.prepare()
        entered, finish = threading.Event(), threading.Event()
        def active():
            with self.runtime._execution_lock:
                entered.set(); finish.wait(5)
        thread = threading.Thread(target=active)
        thread.start(); self.assertTrue(entered.wait(2))
        try:
            with self.assertRaisesRegex(RuntimeError, 'active'):
                self.runtime.control_runtime('reclaim', {})
            self.assertFalse(self.evicted)
        finally:
            finish.set(); thread.join(2)

    def test_core_selection_change_requires_explicit_restart(self):
        self.prepare()
        with self.assertRaisesRegex(RuntimeError, 'restart'):
            self.runtime.SparkTalkUNETLoader().load_unet('other.safetensors', 'default')
        self.assertEqual(len(self.loads), 2)

    def test_static_lora_restores_adapter_without_offloading_core(self):
        self.prepare()
        clone = Patcher(self.runtime._core.model)
        clone.is_dynamic = lambda: False
        self.memory.current_loaded_models.append(types.SimpleNamespace(model=clone))
        self.runtime.control_runtime('reclaim', {})
        self.assertTrue(clone.unpatched)
        self.assertEqual(clone.unpatch_device, clone.load_device)
        self.assertEqual(clone.model.model_loaded_weight_memory, self.runtime._core.model_size())
        self.assertTrue(clone.patches)


class WarmupAPITests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, ready):
        graphs = []
        posts = []
        async def execute(graph): graphs.append(graph); return 'discarded-image'
        def workflow(*args):
            return {'1': {'class_type': 'SparkTalkUNETLoader'}, '2': {'class_type': 'CLIPLoader'},
                    '3': {'class_type': 'SparkTalkVAELoader'}, '4': {'class_type': 'SparkTalkTextEncode'},
                    '8': {'inputs': {'steps': 4}}}
        base = types.SimpleNamespace(generation_lock=asyncio.Lock(), workflow=workflow, execute_workflow=execute,
                                     DIFFUSION_MODEL='core', VAE='vae', COMFY_URL='http://comfy')
        class Response:
            is_success = True
            def __init__(self, prepared): self.prepared = prepared
            def json(self): return {'status': 'ok', 'core_ready': self.prepared}
            def raise_for_status(self): pass
        class Client:
            def __init__(self, **kwargs): self.prepared = ready
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def post(self, url, json): posts.append(url); self.prepared = True; return Response(ready)
            async def get(self, url): return Response(self.prepared)
        # Execute the real API function with an in-memory HTTP transport. No
        # FastAPI dependency, listening port, model or CUDA context is needed.
        tree = ast.parse((SOURCE.parents[1] / 'api.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'runtime_control')
        namespace = {'base': base, 'httpx': types.SimpleNamespace(AsyncClient=Client), 'uuid': uuid, 'HTTPException': RuntimeError}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'runtime-control', 'exec'), namespace)
        state = await namespace['runtime_control']('prepare')
        return state, graphs, posts

    async def test_dynamic_core_uses_one_step_without_text_encoder(self):
        state, graphs, posts = await self.exercise(False)
        self.assertTrue(state['core_ready'])
        self.assertEqual(len(graphs), 1)
        self.assertEqual(posts, ['http://comfy/sparktalk/prepare'])
        self.assertNotIn('2', graphs[0])
        self.assertEqual(graphs[0]['4']['class_type'], 'SparkTalkWarmupConditioning')
        self.assertEqual(graphs[0]['8']['inputs']['steps'], 1)

    async def test_resident_core_does_not_run_another_warmup(self):
        state, graphs, posts = await self.exercise(True)
        self.assertTrue(state['core_ready'])
        self.assertFalse(graphs)
        self.assertFalse(posts)


if __name__ == '__main__': unittest.main()
