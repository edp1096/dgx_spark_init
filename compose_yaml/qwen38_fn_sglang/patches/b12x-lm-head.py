"""Install the qualified BF16 one-row LM-head path in a pinned SGLang image."""
from pathlib import Path
import ast
p=Path('/sgl-workspace/sglang/python/sglang/srt/layers/logits_processor.py')
s=p.read_text()
anchor='        elif hasattr(lm_head, "weight"):\n            # Normal linear layer'
assert s.count(anchor)==1, 'SGLang LM-head source changed; review before patching'
replacement='        elif hasattr(lm_head, "weight"):\n            # Narrow opt-in: preserve BF16 weights/logits and all outer TP logic.\n            if (not self.use_fp32_lm_head and self.rl_on_policy_target is None\n                and hidden_states.dtype == torch.bfloat16\n                and lm_head.weight.dtype == torch.bfloat16\n                and hidden_states.shape == (1, 2560)\n                and lm_head.weight.shape == (124160, 2560)):\n                from b12x.gemm.bf16_gemv import mm\n                if not getattr(self, "_b12x_probe_head_logged", False):\n                    print("B12X LM head active: BF16 1x2560 by 124160x2560", flush=True)\n                    self._b12x_probe_head_logged = True\n                return mm(hidden_states, lm_head.weight, output_dtype=torch.bfloat16)\n            # Normal linear layer'
updated=s.replace(anchor,replacement)
ast.parse(updated)
p.write_text(updated)
print('Installed BF16 one-row B12X LM head; weights/logits remain BF16')
