"""Preserve Gemma proportional position interpolation when selecting >256K."""
from pathlib import Path
p=Path('/sgl-workspace/sglang/python/sglang/srt')
f=p/'models/gemma4_causal.py';s=f.read_text();old='rope_scaling={"rope_type": rope_parameters.get("rope_type", "default")},';new='rope_scaling={"rope_type": rope_parameters.get("rope_type", "default"), "factor": rope_parameters.get("factor", 1.0)},'
if new not in s:
 assert s.count(old)==1;s=s.replace(old,new);f.write_text(s)
f=p/'layers/rotary_embedding/rope_variant.py';s=f.read_text();i=s.index('class Gemma4RotaryEmbedding(');before,part=s[:i],s[i:]
if 'self.scaling_factor' not in part:
 old='        dtype: torch.dtype,\n    ) -> None:';assert part.count(old)==1
 part=part.replace(old,'        dtype: torch.dtype,\n        scaling_factor: float = 1.0,\n    ) -> None:')
 part=part.replace('        # Store angles before calling super().__init__','        if not 1 <= scaling_factor <= 4:\n            raise ValueError("Invalid Gemma position factor")\n        self.scaling_factor = scaling_factor\n        # Store angles before calling super().__init__')
 old='        inv_freq = 1.0 / (base**freq_exponents)';assert part.count(old)==1;part=part.replace(old,old+' / self.scaling_factor');f.write_text(before+part)
f=p/'layers/rotary_embedding/factory.py';s=f.read_text();i=s.index('        elif scaling_type == "proportional":');j=s.index('        else:',i);part=s[i:j]
if 'scaling_factor=' not in part:
 old='                dtype,\n';assert part.count(old)==1;part=part.replace(old,old+'                scaling_factor=rope_scaling.get("factor", 1.0),\n');f.write_text(s[:i]+part+s[j:])
print('Gemma proportional factor enabled')
