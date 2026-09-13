"""Test-only comparison of upstream's replacement recipe against original outputs.

Never imported by serving. Upstream no longer accepts numerical_recipe, so this
explicitly probes its default computation to quantify any resulting difference.
The underlying check still requires identical packed bytes and numerical outputs.
"""
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from b12x.moe import fused_moe
import b12x_slots

original_spec = fused_moe.ActivationSpec


def upstream_spec(**kwargs):
    assert kwargs.pop('numerical_recipe') == 'deepseek_v41'
    return original_spec(**kwargs)


fused_moe.ActivationSpec = upstream_spec
original_buffers = b12x_slots.execution_buffers


def upstream_buffers(m, topk, shared):
    buffers = original_buffers(m, topk, shared)
    # Upstream also removed the original FP32 output boundary. Keep that
    # change explicit and confined to this probe, then measure the error.
    if buffers['out'].dtype != torch.bfloat16:
        buffers['out'] = torch.empty_like(buffers['out'], dtype=torch.bfloat16)
    return buffers


b12x_slots.execution_buffers = upstream_buffers
print('PROBE ONLY: upstream default recipe; not a precision-preserving serving change', flush=True)
runpy.run_path(str(Path(__file__).with_name('check_b12x_upgrade.py')), run_name='__main__')
