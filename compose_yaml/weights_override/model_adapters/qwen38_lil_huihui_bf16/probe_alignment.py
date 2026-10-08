"""Check source/native HF matrix alignment against the reconstructed QAD base."""
import argparse
from pathlib import Path
import torch
from weights_core.safetensors_io import Checkpoint
from quantize import decode
from probe_numeric import source
from fetch_audit import atomic_json


def cosine(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(a.dot(b)/(a.norm()*b.norm()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--audit-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    cp = Checkpoint(a.base)
    results = []
    for layer in [0, 1]:
        prefix = f'model.language_model.layers.{layer}.linear_attn.out_proj'
        original = source(a.audit_root/'original'/(prefix+'.weight.bf16'), [2560, 6144])
        base, _ = decode(cp, prefix)
        native = cosine(base, original)
        permuted = cosine(base, original.reshape(2560, 16, 3, 128).transpose(1, 2).reshape(2560, 6144))
        row = dict(tensor=prefix, native_cosine=native, gguf_order_cosine=permuted)
        results.append(row)
        if native < .9 or native < permuted+.2:
            raise ValueError(('HF output projection alignment failed', row))
        name = f'model.language_model.layers.{layer}.mlp.experts.down_proj'
        original = source(a.audit_root/'original'/(name+'.bf16'), [2560, 640])
        base, _ = decode(cp, f'model.language_model.layers.{layer}.mlp.experts.0.down_proj')
        value = cosine(base, original)
        results.append(dict(tensor=name+'.expert_0', native_cosine=value))
        if value < .9:
            raise ValueError(('Expert source alignment failed', value))
    atomic_json(a.output, dict(status='passed', results=results))
    print(results, flush=True)
