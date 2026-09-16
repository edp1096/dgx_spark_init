"""CPU check against the installed Transformers proportional RoPE reference."""
import json,sys
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
import vllm.model_executor.layers.rotary_embedding as rotary
from vllm.model_executor.layers.rotary_embedding.gemma4_rope import Gemma4RotaryEmbedding

# Test the actual frequency method and factory forwarding without allocating a
# CustomOp outside a live engine (which requires CUDA engine configuration).
class CPUFrequencyView:
    _compute_inv_freq = Gemma4RotaryEmbedding._compute_inv_freq
    def __init__(self,head_size,rotary_dim,max_position,base,is_neox_style,dtype,scaling_factor=1.0):
        self.head_size=head_size;self.rope_angles=rotary_dim//2
        self.nope_angles=head_size//2-self.rope_angles;self.scaling_factor=scaling_factor
rotary.Gemma4RotaryEmbedding=CPUFrequencyView

config=json.load(open(sys.argv[1]))['text_config']
for factor in (1.0,2.0,4.0):
 value=json.loads(json.dumps(config));value['rope_parameters']['full_attention']['factor']=factor
 hf=Gemma4TextConfig(**value)
 reference,_=ROPE_INIT_FUNCTIONS['proportional'](hf,device=torch.device('cpu'),layer_type='full_attention')
 rope=rotary.get_rope(config['global_head_dim'],max_position=16,rope_parameters=value['rope_parameters']['full_attention'],is_neox_style=True,dtype=torch.float32)
 actual=rope._compute_inv_freq(value['rope_parameters']['full_attention']['rope_theta'])
 torch.testing.assert_close(actual,reference,rtol=1e-7,atol=0)
 assert torch.count_nonzero(actual[64:])==0
 print('factor',factor,'matches HF; unrotated dimensions stay zero',flush=True)
