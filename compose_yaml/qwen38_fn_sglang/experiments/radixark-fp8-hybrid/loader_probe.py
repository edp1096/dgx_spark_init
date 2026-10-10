"""Prove expert indexing preserves the original matches and CPU clone bytes."""
import json
import time
import torch
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
import loader_hooks

entries=FusedMoE.make_expert_params_mapping('gate_proj','down_proj','up_proj',512)
names=[f'model.layers.3.mlp.experts.{expert}.{projection}.{suffix}'
       for expert in range(512) for projection in ['gate_proj','up_proj','down_proj']
       for suffix in ['weight','weight_scale','weight_scale_2','input_scale']]
names += ['model.layers.1.self_attn.q_proj.weight','model.layers.1.mlp.shared_expert.gate_proj.weight',
          'model.layers.1.mlp.experts.512.gate_proj.weight','mtp.layers.0.mlp.experts.gate_up_proj.weight']
started=time.monotonic()
expected=[[entry for entry in entries if entry[1] in name] for name in names]
legacy=time.monotonic()-started
started=time.monotonic()
actual=[loader_hooks.candidates(name,entries) for name in names]
indexed=time.monotonic()-started
assert actual==expected
loader_hooks.install()
from sglang.srt.model_loader import loader
from safetensors import safe_open
from pathlib import Path
root=Path('/hf/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4')
shard=root/'layer-00000-experts-0000-0127.safetensors'
for i,(name,tensor) in enumerate(loader.buffered_multi_thread_safetensors_weights_iterator([str(shard)],max_workers=1)):
    with safe_open(shard,framework='pt',device='cpu') as f:
        original=f.get_tensor(name)
        assert torch.equal(tensor.reshape(-1).view(torch.uint8),original.reshape(-1).view(torch.uint8))
        assert tensor.data_ptr()!=original.data_ptr()
    if i>=31:break
print(json.dumps(dict(mapping_cases=len(names),byte_checked_tensors=i+1,legacy_seconds=legacy,indexed_seconds=indexed)),flush=True)
