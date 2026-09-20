"""Exercise the real Qwen4 load_weights method on a tiny fused projection."""
from types import SimpleNamespace
import torch
from torch import nn
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration as Qwen
class Model(nn.Module):
    _load_qwen4_exp_ple_buffer=Qwen._load_qwen4_exp_ple_buffer
    load_weights=Qwen.load_weights
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(tie_word_embeddings=False,encoder_only=False)
        self.start_layer=0;self.end_layer=48;self.language_model_only=True
        self.model=nn.Module();self.model.layers=nn.ModuleList([nn.Module()])
        self.model.layers[0].linear_attn=nn.Module()
        linear=nn.Module();self.model.layers[0].linear_attn.in_proj_ba=linear
        linear.register_parameter('weight_scale_inv',nn.Parameter(torch.full((4,2),255,dtype=torch.uint8),requires_grad=False))
        linear.weight_scale_inv.weight_loader=lambda p,w,part:p.data[part*2:(part+1)*2].copy_(w)
model=Model()
a=torch.tensor([[120,121],[122,123]],dtype=torch.uint8)
b=torch.tensor([[124,125],[126,127]],dtype=torch.uint8)
model.load_weights([('model.language_model.layers.0.linear_attn.in_proj_a.weight_scale',a),('model.language_model.layers.0.linear_attn.in_proj_b.weight_scale',b)])
actual=model.model.layers[0].linear_attn.in_proj_ba.weight_scale_inv
assert torch.equal(actual,torch.cat((b,a)))
print('Checkpoint MXFP8 weight_scale reaches both SGLang fused weight_scale_inv partitions exactly')
