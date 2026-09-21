"""Order all router input reads after the PDL producer dependency.

Qwen's softmax router supplies a freshly allocated zero bias. Prefetching it
before gdc_wait can read old allocation contents while the producer is still
running. Keep PDL and the numerical algorithm; move the bias load after the
wait. Applied only to the QAD TP1 image, not the legacy/TP2 target.
"""
import ast
from pathlib import Path

path = Path('/sgl-workspace/sglang/python/sglang/kernels/ops/moe/moe_fused_gate.py')
source = path.read_text()
old = '''    # prefetch bias before PDL wait
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )  # [BLOCK_N]

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
'''
new = '''    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    # Bias can be freshly produced (Qwen's zero bias), just like scores.
    # Both inputs must be read only after the PDL producer dependency.
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )  # [BLOCK_N]
'''
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError('Unexpected router source; re-audit PDL ordering patch')
    source = source.replace(old, new, 1)
    ast.parse(source)
    path.write_text(source)
print('QAD TP1 router waits for its bias producer; PDL remains enabled')
