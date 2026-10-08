"""Match Ornith/Qwen GDN warmup to the loaded normalization weight."""
import importlib.util
from pathlib import Path
root=Path(importlib.util.find_spec("vllm").origin).parent

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
