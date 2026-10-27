"""PDMD 4-NFE residuals over the unchanged MiniMax H3 NVFP4 model.

Source: https://huggingface.co/Iwannapose/minimax_h3_pdmd_4nfe_comfyui
Revision ae6ab62d28754b9ce052798b92d9567161568f28 (v6, BF16).
Apply x @ A.T @ B.T after each original linear; do not requantize the delta.
"""
import gc
import torch
import torch.nn.functional as F
from safetensors import safe_open

FILENAME = 'minimax_h3_pdmd_4nfe_comfyui_v6.safetensors'
REVISION = 'ae6ab62d28754b9ce052798b92d9567161568f28'
SHA256 = 'd1d7c9f08782492ebcaa57f9f823899541ee04c48a298692a229683c4b3dd8a1'


class PDMDAdapter:
    def __init__(self, model, path):
        self.handles = []
        self.calls = {}
        self.weights = []
        self.bytes = 0
        try:
            with safe_open(path, framework='pt', device='cpu') as source:
                keys = set(source.keys())
                names = sorted(k.removesuffix('.lora_A.weight') for k in keys if k.endswith('.lora_A.weight'))
                expected = {n + suffix for n in names for suffix in ('.lora_A.weight', '.lora_B.weight', '.alpha')}
                if len(names) != 208 or keys != expected:
                    raise ValueError('PDMD v6 must contain exactly 208 complete LoRA pairs and alpha values')
                for name in names:
                    module = model.model.get_submodule(name)
                    a = source.get_tensor(name + '.lora_A.weight').to('cuda')
                    b = source.get_tensor(name + '.lora_B.weight').to('cuda')
                    scale = float(source.get_tensor(name + '.alpha')) / a.shape[0]
                    if (scale != 1.0 or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16
                            or a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]
                            or tuple(module.weight.shape) != (b.shape[0], a.shape[1])
                            or getattr(module, 'pre_quant_scale', None) is not None):
                        raise ValueError('PDMD v6 does not match the qualified base layer: ' + name)
                    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                        raise ValueError('Nonfinite PDMD adapter: ' + name)
                    self.bytes += (a.numel() + b.numel()) * a.element_size()
                    self.calls[name] = 0
                    self.weights.append((module, module.weight))

                    def hook(_module, args, output, a=a, b=b, name=name):
                        x = args[0]
                        if x.dtype != a.dtype or x.device != a.device:
                            raise RuntimeError('Unexpected PDMD activation dtype/device: ' + name)
                        self.calls[name] += 1
                        return output + F.linear(F.linear(x, a), b)

                    self.handles.append(module.register_forward_hook(hook))
            torch.cuda.synchronize()
            gc.collect()
        except Exception:
            self.remove()
            raise

    def report(self):
        if any(module.weight is not weight for module, weight in self.weights):
            raise RuntimeError('PDMD base weights changed after installation')
        return {'method': 'bf16_residual', 'revision': REVISION, 'sha256': SHA256,
                'modules': len(self.calls), 'called_modules': sum(n > 0 for n in self.calls.values()),
                'adapter_bytes': self.bytes, 'base_weights_unchanged': True}

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.weights.clear()
