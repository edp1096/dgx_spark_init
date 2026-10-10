"""Opt-in FP8 side-layer dispatch for the pinned SGLang ModelOpt NVFP4 runtime."""
import json
import os
from pathlib import Path


def install():
    model_path = os.environ.get('RADIXARK_FP8_HYBRID_PATH')
    if not model_path:
        return
    manifest = json.loads((Path(model_path) / 'fp8-hybrid-manifest.json').read_text())
    assert len(manifest['tensors']) == 300
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
    if getattr(ModelOptFp4Config, '_radixark_fp8_hybrid_installed', False):
        return

    def normalize(name):
        return name[name.index('layers.'):] if 'layers.' in name else name

    selected = {normalize(name[:-len('.weight')]) for name in manifest['tensors']}
    original = ModelOptFp4Config.get_quant_method

    def method(self, layer, prefix):
        if isinstance(layer, LinearBase) and not prefix.startswith('mtp.'):
            head, _, leaf = prefix.rpartition('.')
            fused = (self.packed_modules_mapping or {}).get(leaf)
            names = [f'{head}.{part}' for part in fused] if fused else [prefix]
            hits = [normalize(name) in selected for name in names]
            if any(hits) and not all(hits):
                raise ValueError(f'Partially converted fused layer: {prefix}: {names}')
            if hits and all(hits):
                config = Fp8Config(is_checkpoint_fp8_serialized=True,
                                   activation_scheme='dynamic', weight_block_size=[128, 128])
                result = config.get_quant_method(layer, prefix)
                native_prepare = result.process_weights_after_loading
                def prepare(module):
                    import torch
                    scale = module.weight_scale_inv
                    if module.weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
                        raise ValueError(f'Wrong hybrid storage dtype: {prefix}')
                    if not torch.isfinite(scale).all().item() or not (scale > 0).all().item():
                        raise ValueError(f'Missing or invalid FP8 block scales: {prefix}')
                    native_prepare(module)
                    print(f'RADIXARK_FP8_VERIFIED {prefix}', flush=True)
                result.process_weights_after_loading = prepare
                print(f'RADIXARK_FP8_LAYER {prefix} {type(result).__name__}', flush=True)
                return result
        return original(self, layer, prefix)

    ModelOptFp4Config.get_quant_method = method
    ModelOptFp4Config._radixark_fp8_hybrid_installed = True
    print(f'RADIXARK_FP8_HYBRID enabled tensors={len(selected)}', flush=True)
