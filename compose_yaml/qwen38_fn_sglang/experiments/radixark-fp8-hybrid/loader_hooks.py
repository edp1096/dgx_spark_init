"""Opt-in, fail-on-drift loader improvements; arithmetic and GPU kernels unchanged."""
import inspect
import re
import textwrap
import time

_indices = {}
EXPERT = re.compile(r'experts\.\d+\.[^.]+\.')


def candidates(name, entries):
    key = id(entries)
    if key not in _indices:
        _indices[key] = (entries, {entry[1]: [entry] for entry in entries})
    match = EXPERT.search(name)
    return _indices[key][1].get(match[0], []) if match else []


def install():
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
    cls = Qwen4ExpForConditionalGeneration
    if getattr(cls, '_radixark_fast_loader', False):return
    original = cls.load_weights
    source = textwrap.dedent(inspect.getsource(original))
    anchor = 'for mapping in current_expert_params_mapping:'
    assert source.count(anchor) == 1, 'Pinned Qwen loader changed'
    source = source.replace(anchor, 'for mapping in (current_expert_params_mapping if is_fused_expert else radixark_expert_candidates(name, current_expert_params_mapping)):')
    namespace = dict(original.__globals__, radixark_expert_candidates=candidates)
    exec(compile(source, '<radixark-indexed-loader>', 'exec'), namespace)
    cls.load_weights = namespace['load_weights']
    cls._radixark_fast_loader = True

    from sglang.srt.model_loader import loader
    native = loader.buffered_multi_thread_safetensors_weights_iterator
    def iterator(*args, **kwargs):
        started=time.monotonic();count=0;size=0
        for name,tensor in native(*args,**kwargs):
            if '.mlp.experts.' in name and tensor.device.type=='cpu' and tensor.nbytes <= 64*1024**2:
                tensor=tensor.clone();count+=1;size+=tensor.nbytes
            yield name,tensor
        print(f'RADIXARK_FAST_LOADER tensors={count} bytes={size} seconds={time.monotonic()-started:.3f}',flush=True)
    loader.buffered_multi_thread_safetensors_weights_iterator=iterator
