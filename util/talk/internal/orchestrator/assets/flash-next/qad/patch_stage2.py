"""Scoped vision/MTP/B12X changes on the already-qualified trial image."""
import ast
from pathlib import Path

def patch(path, old, new):
    p=Path('/sgl-workspace/sglang/python/sglang/srt')/path
    s=p.read_text()
    if s.count(old)!=1:
        raise RuntimeError(f'Unexpected pinned source anchor: {path}: {old[:100]}')
    s=s.replace(old,new)
    ast.parse(s)
    p.write_text(s)

patch('models/qwen3_vl.py',
      '                quant_config=None,\n                norm_eps=',
      '                quant_config=quant_config if quant_config is not None and quant_config.get_name() == "modelopt_mixed" else None,\n                norm_eps=')
patch('layers/quantization/modelopt_quant.py',
      '        candidates = [prefix]\n',
      '''        candidates = [prefix]
        if prefix.startswith("model.visual.") and prefix.endswith(".attn.qkv_proj"):
            candidates.append(prefix[:-len("qkv_proj")] + "qkv")
''')
p=Path('/sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py')
s=p.read_text()
start=s.index('    def process_weights_after_loading(',s.index('class ModelOptNvFp4A16LinearMethod'))
end=s.index('\n\ndef _compute_gemm1_alphas',start)
s=s[:start]+'''    def process_weights_after_loading(self, layer):
        from qad_b12x import prepare_a16_linear
        prepare_a16_linear(layer)

    def apply(self, layer, x, bias=None):
        from qad_b12x import apply_a16_linear
        return apply_a16_linear(layer, x, bias)
'''+s[end:]
ast.parse(s);p.write_text(s)
print('Installed mixed-precision vision and BF16-activation B12X W4A16 adapter')
patch('layers/logits_processor.py',
'''        elif hasattr(lm_head, "weight"):
            # Normal linear layer''',
'''        elif hasattr(lm_head, "weight"):
            if (not self.use_fp32_lm_head and self.rl_on_policy_target is None
                and embedding_bias is None and hidden_states.ndim == 2
                and 1 <= hidden_states.shape[0] <= 4
                and hidden_states.shape[1] == 2560
                and hidden_states.dtype == torch.bfloat16
                and lm_head.weight.dtype == torch.bfloat16
                and lm_head.weight.shape == (248320, 2560)):
                from b12x.gemm.bf16_gemv import mm
                return mm(hidden_states.contiguous(), lm_head.weight, output_dtype=torch.bfloat16)
            # Normal linear layer''')
patch('entrypoints/openai/serving_chat.py',
'''        """Process chat messages and apply chat template"""
        if self.default_chat_template_kwargs:''',
'''        """Process chat messages and apply chat template"""
        if (request.reasoning_effort == "none" and self.reasoning_parser == "qwen3"
            and self.tokenizer_manager.model_config.hf_config.model_type == "qwen4_exp"):
            # Keep template rendering and reasoning parsing in agreement for
            # existing OpenAI clients, including Talk's qwen3.8 profile.
            request.chat_template_kwargs = dict(request.chat_template_kwargs or {})
            request.chat_template_kwargs.setdefault("enable_thinking", False)
            request.reasoning_effort = None
        if self.default_chat_template_kwargs:''')
# Restrict the dense override to MXFP8; keep every other FP8 format on its
# original implementation. Preserve dynamic MXFP8 activation quantization.
p=Path('/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py')
s=p.read_text();start=s.index('class Fp8LinearMethod');end=s.index('\nclass ',start+1)
part=s[start:end]
old='    def process_weights_after_loading(self, layer: Module) -> None:\n'
assert part.count(old)==1
part=part.replace(old,old+'''        if self.use_mxfp8:
            from qad_b12x import prepare_mxfp8_linear
            prepare_mxfp8_linear(layer)
            return
''')
old='''    ) -> torch.Tensor:
        if self.use_marlin:'''
assert part.count(old)==1
part=part.replace(old,'''    ) -> torch.Tensor:
        if hasattr(layer, "_qad_mxfp8"):
            from qad_b12x import apply_mxfp8_linear
            return apply_mxfp8_linear(layer, x, bias)
        if self.use_marlin:''')
s=s[:start]+part+s[end:];ast.parse(s);p.write_text(s)
patch('layers/moe/fused_moe_triton/layer.py',
      '        if "ModelOpt" in method.__class__.__name__:',
      '        if isinstance(method, ModelOptNvFp4FusedMoEMethod) or "ModelOpt" in method.__class__.__name__:')
patch('layers/quantization/modelopt_quant.py',
'''            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoEMethod(self.nvfp4_config)''',
'''            if quant_algo == "NVFP4":
                from qad_moe import QADNVFP4MoEMethod
                return QADNVFP4MoEMethod(self.nvfp4_config)''')
patch('layers/quantization/modelopt_quant.py',
'''            if quant_algo == "W4A16_NVFP4":
                return ModelOptNvFp4FusedMoEMethod(self.nvfp4a16_config)''',
'''            if quant_algo == "W4A16_NVFP4":
                from qad_moe import QADW4A16MoEMethod
                return QADW4A16MoEMethod(self.nvfp4a16_config)''')
patch('models/qwen4_exp_mtp.py',
      '        quant_config = _mtp_quant_config(quant_config)',
'''        has_quantized_mtp = quant_config is not None and any(
            key.startswith("mtp.") for key in getattr(quant_config,"quantized_layers",{})
        )
        if not has_quantized_mtp:
            quant_config = _mtp_quant_config(quant_config)''')
patch('layers/quantization/fp8_utils.py',
'''    if backend == "cutlass" and original_n < 128:
        padded = torch.zeros((128, weight.shape[1]), dtype=weight.dtype, device=weight.device)''',
'''    if backend == "cutlass" and original_n % 128:
        padded_n = (original_n + 127) // 128 * 128
        padded = torch.zeros((padded_n, weight.shape[1]), dtype=weight.dtype, device=weight.device)''')
