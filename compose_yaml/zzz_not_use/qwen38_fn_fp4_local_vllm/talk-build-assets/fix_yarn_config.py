"""Expose text context length before Transformers 5 normalizes outer YaRN."""
import ast
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/config.py')
source = path.read_text()
needle = '''        self.rope_parameters = rope_parameters or getattr(
            self.text_config, "rope_parameters", {}
        )
'''
replacement = '''        # Transformers normalizes outer RoPE in PretrainedConfig.__post_init__
        # before assigning additional kwargs. YaRN needs this attribute then.
        self.max_position_embeddings = self.text_config.max_position_embeddings
''' + needle
if source.count(needle) != 1:
    raise RuntimeError('Pinned Qwen composite-config patch no longer matches')
source = source.replace(needle, replacement)
ast.parse(source)
path.write_text(source)
print('Qwen composite YaRN context propagation installed')

# MRoPE reserves four times the requested positions for video timestamps.
# That storage reservation must not change the original training length used
# to select YaRN's interpolation/extrapolation frequency ranges.
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/rotary_embedding/mrope.py')
source = path.read_text()
needle = '        self.cache_max_position_num = max_position_embeddings * 4\n'
if source.count(needle) != 1:
    raise RuntimeError('Pinned MRoPE cache initialization no longer matches')
source = source.replace(needle, '        self._yarn_original_max_position_embeddings = max_position_embeddings\n' + needle)
needle = '        return YaRNScalingRotaryEmbedding._compute_inv_freq(self, base)\n'
replacement = '''        from types import SimpleNamespace
        parameters = SimpleNamespace(
            base=self.base, rotary_dim=self.rotary_dim,
            beta_fast=self.beta_fast, beta_slow=self.beta_slow,
            max_position_embeddings=self._yarn_original_max_position_embeddings,
            truncate=self.truncate,
        )
        return YaRNScalingRotaryEmbedding._compute_inv_freq(parameters, base)
'''
if source.count(needle) != 1:
    raise RuntimeError('Pinned MRoPE YaRN frequency function no longer matches')
source = source.replace(needle, replacement)
ast.parse(source)
path.write_text(source)
print('MRoPE YaRN training length separated from video cache capacity')
