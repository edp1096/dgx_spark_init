"""Make vLLM Gemma4 proportional RoPE honor the factor supported by HF.

The baseline vLLM constructor ignores rope_parameters.factor. Preserve the
head_dim denominator and zero-padded unrotated dimensions; divide only the
existing frequencies by factor, matching HF proportional RoPE. A 1M context
still requires long-context quality validation, beyond the native 262144.
"""
import importlib.util
from pathlib import Path
root=Path(importlib.util.find_spec('vllm').origin).parent
p=root/'model_executor/layers/rotary_embedding/gemma4_rope.py'
s=p.read_text()
if 'self.scaling_factor' not in s:
 if 'SPARKTALK_GEMMA_POSITION_FACTOR' in s:raise RuntimeError('Rebuild from the clean pinned base')
 parameter='        dtype: torch.dtype,\n    ) -> None:'
 anchor='        inv_freq = 1.0 / (base**freq_exponents)'
 if s.count(parameter)!=1 or s.count(anchor)!=1:raise RuntimeError('Unexpected Gemma4 RoPE source')
 s=s.replace('import torch','import math\nimport torch',1).replace(parameter,'        dtype: torch.dtype,\n        scaling_factor: float = 1.0,\n    ) -> None:')
 s=s.replace('        self.rope_angles = rotary_dim // 2','''        if not math.isfinite(scaling_factor) or not 1 <= scaling_factor <= 4:
            raise ValueError("Invalid Gemma4 position interpolation factor")
        self.scaling_factor = scaling_factor
        self.rope_angles = rotary_dim // 2''')
 s=s.replace(anchor,anchor+'\n        inv_freq = inv_freq / self.scaling_factor')
 p.write_text(s)
p=root/'model_executor/layers/rotary_embedding/__init__.py';s=p.read_text()
start=s.index('    elif scaling_type == "proportional":');end=s.index('    elif scaling_type == "llama3":',start)
part=s[start:end]
if 'scaling_factor=' not in part:
 anchor='            dtype,\n        )'
 if part.count(anchor)!=1:raise RuntimeError('Unexpected proportional RoPE factory')
 part=part.replace(anchor,'            dtype,\n            scaling_factor=rope_parameters.get("factor", 1.0),\n        )');s=s[:start]+part+s[end:];p.write_text(s)
print('Gemma4 proportional factor matches HF configuration semantics')

# Qwen3.5 normalizes per head; Qwen3-Next may normalize the full feature vector.
# Match the actual norm weight rather than assuming hv*v in warmup-only inputs.
p=root/'model_executor/warmup/qwen_triton_warmup.py';s=p.read_text()
if 'rows_per_token = ' not in s:
 old='''    feature_size = int(config.hv * config.v)
    group_size = min(int(config.norm_group_size), feature_size)
    lengths = (1, 2, 16, 32, 128, 1024)
    for length in lengths:
        x = torch.empty((length, feature_size), dtype=config.conv_dtype, device=device)'''
 new='''    feature_size = int(config.norm_weight.numel())
    total_features = int(config.hv * config.v)
    if feature_size <= 0 or total_features % feature_size:
        raise ValueError("Unexpected Qwen GDN norm weight shape")
    rows_per_token = total_features // feature_size
    group_size = min(int(config.norm_group_size), feature_size)
    lengths = (1, 2, 16, 32, 128, 1024)
    for length in lengths:
        x = torch.empty((length * rows_per_token, feature_size), dtype=config.conv_dtype, device=device)'''
 if s.count(old)!=1:raise RuntimeError('Unexpected Qwen GDN norm warmup source')
 p.write_text(s.replace(old,new))
print('Qwen GDN norm warmup matches loaded normalization weight')
