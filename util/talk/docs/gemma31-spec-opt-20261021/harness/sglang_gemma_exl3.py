"""Opt-in GPU EXL3 matrices for SGLang's Gemma 4 31B Frozen-KV assistant.

Only the assistant loader/matrices change. Target NVFP4 execution, attention,
sampling, verification, streaming and tool parsing remain owned by SGLang.
"""
import logging
import os

import torch
from torch import nn

logger = logging.getLogger(__name__)


class Matrix(nn.Module):
    def __init__(self, weights, name, k, n, head=False):
        super().__init__()
        from exllamav3.modules.quant.exl3 import LinearEXL3
        self.k, self.n, self.head = k, n, head
        for suffix in ['trellis', 'suh', 'svh', 'mul1']:
            key = name + '.' + suffix
            tensor = weights.pop(key).contiguous().to('cuda')
            self.register_buffer(suffix, tensor)
        if self.suh.numel() != k or self.svh.numel() != n:
            raise ValueError('EXL3 matrix geometry mismatch: ' + name)
        bits = self.trellis.shape[-1] // 16
        if bits != (6 if head else int(os.environ["ASSISTANT_BODY_BITS"])):
            raise ValueError('Expected declared body bits/head6 EXL3 assistant: ' + name)
        self.inner = LinearEXL3(None, k, n, suh=self.suh, svh=self.svh,
                                trellis=self.trellis, mul1=self.mul1,
                                out_dtype=torch.float16, key=name)
        # LogitsProcessor already has a path for a head exposing quant_method
        # instead of a dense .weight. No dequantized head is kept on the GPU.
        if head:
            self.quant_method = HeadMethod()

    def compute(self, x, output_dtype=None):
        original_dtype = x.dtype
        x = x.to(torch.float16).contiguous()
        output = self.inner.forward(x, {}, out_dtype=output_dtype)
        return output if output_dtype is not None else output.to(original_dtype)

    def forward(self, x):
        return self.compute(x), None


class HeadMethod:
    def process_weights_after_loading(self, layer):
        # EXL3Matrix construction already binds the packed CUDA operands.
        return

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError('Assistant EXL3 head does not use bias')
        return layer.compute(x, output_dtype=torch.float32)


class QueryProjection(nn.Module):
    def __init__(self, query, kv_size):
        super().__init__()
        self.query = query
        self.kv_size = kv_size

    def forward(self, x):
        # The original attention wrapper splits a QKV tensor, then discards
        # K/V when it is bound to the target's frozen caches.
        q = self.query.compute(x)
        return torch.nn.functional.pad(q, (0, 2 * self.kv_size)), None


class GateUp(nn.Module):
    def __init__(self, gate, up):
        super().__init__()
        self.gate, self.up = gate, up

    def forward(self, x):
        return torch.cat((self.gate.compute(x), self.up.compute(x)), dim=-1), None


def install():
    if os.environ.get('SPARKTALK_GEMMA_ASSISTANT_EXL3') != '1':
        return
    from sglang.srt.models.gemma4_mtp import Gemma4AssistantForCausalLM
    if getattr(Gemma4AssistantForCausalLM, '_exl3_installed', False):
        return
    original_load = Gemma4AssistantForCausalLM.load_weights

    def load_weights(self, iterator):
        from sglang.srt.server_args import get_global_server_args
        args = get_global_server_args()
        if args.tp_size != 1 or args.max_running_requests != 1:
            raise ValueError('Gemma assistant EXL3 is qualified only for TP1 / one running request')
        if self.hidden_size != 1024 or self.backbone_hidden_size != 5376 or self.use_ordered_embeddings:
            raise ValueError('Expected the standard Gemma 4 31B assistant')
        weights = dict(iterator)
        if 'lm_head.trellis' not in weights:
            return original_load(self, weights.items())
        original_keys = set(weights)
        norms = {k: v for k, v in weights.items() if k.endswith('.weight') or k.endswith('.layer_scalar')}
        original_load(self, norms.items())
        for name in norms:
            weights.pop(name)
        self.pre_projection = Matrix(weights, 'pre_projection', 10752, 1024)
        self.post_projection = Matrix(weights, 'post_projection', 1024, 5376)
        for index, layer in enumerate(self.model.layers):
            name = f'model.layers.{index}'
            attn = layer.self_attn
            query = Matrix(weights, name + '.self_attn.q_proj', 1024, attn.q_size)
            attn.qkv_proj = QueryProjection(query, attn.kv_size)
            attn.o_proj = Matrix(weights, name + '.self_attn.o_proj', attn.q_size, 1024)
            gate = Matrix(weights, name + '.mlp.gate_proj', 1024, 8192)
            up = Matrix(weights, name + '.mlp.up_proj', 1024, 8192)
            layer.mlp.gate_up_proj = GateUp(gate, up)
            layer.mlp.down_proj = Matrix(weights, name + '.mlp.down_proj', 8192, 1024)
        self.lm_head = Matrix(weights, 'lm_head', 1024, self.vocab_size, head=True)
        # Assistant input embeddings come from the target via set_embed_and_head.
        self.model.embed_tokens = nn.Identity()
        if weights:
            raise ValueError('Unconsumed assistant tensors: ' + ', '.join(sorted(weights)))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info('GEMMA_ASSISTANT_EXL3_READY matrices=23 body_bits=%s head_bits=6 gpu=1 dense_head=0', os.environ['ASSISTANT_BODY_BITS'])
        return original_keys

    Gemma4AssistantForCausalLM.load_weights = load_weights
    Gemma4AssistantForCausalLM._exl3_installed = True
