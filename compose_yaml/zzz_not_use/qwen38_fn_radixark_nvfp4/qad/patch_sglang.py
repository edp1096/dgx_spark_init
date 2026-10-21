"""Apply narrowly scoped NVFP4 PLE integration to the pinned Apache-2.0 source."""
from pathlib import Path
import ast
p=Path('/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py')
s=p.read_text()
def replace(old,new):
 global s
 if s.count(old)!=1:raise RuntimeError(f'Pinned SGLang anchor changed: {old[:80]!r}')
 s=s.replace(old,new)
replace('import math\n','import math\nfrom ple_embedding import NVFP4PLEEmbedding\n')
start=s.index('        self.ngram_embedding = VocabParallelEmbedding(')
end=s.index('\n\n    @classmethod',start)
original=s[start:end]
new='''        if getattr(config, "ple_embedding_dtype", None) == "nvfp4":
            if not config.ple_offload_embedding or getattr(config,"ple_offload_backend",None)!="file":
                raise ValueError("NVFP4 PLE trial requires file offload")
            if get_tp_group().world_size != 1:
                raise ValueError("NVFP4 PLE trial supports TP1 only")
            self.ngram_embedding = NVFP4PLEEmbedding(
                padded_vocab_size, self.head_dim_per_ngram,
                int(config.split_ngram_parts), config.ple_offload_dir)
        else:
'''+ '\n'.join('    '+line for line in original.splitlines())
s=s[:start]+new+s[end:]
replace('        if config.ple_offload_embedding:\n            self.ple_embedding.ngram_embedding = Qwen4ExpPinnedHostEmbedding(',
'''        if config.ple_offload_embedding and not isinstance(self.ple_embedding.ngram_embedding, NVFP4PLEEmbedding):
            self.ple_embedding.ngram_embedding = Qwen4ExpPinnedHostEmbedding(''')
anchor='''            if self._load_qwen4_exp_ple_buffer(
                name, loaded_weight, buffers, loaded_buffers
            ):'''
replace(anchor,'''            if ".ngram_embedding." in name:
                embedding_name, suffix = name.split(".ngram_embedding.", 1)
                owner = ple_modules.get(embedding_name)
                if owner is not None and isinstance(owner.ngram_embedding, NVFP4PLEEmbedding):
                    if not owner.ngram_embedding.load_tensor(suffix, loaded_weight):
                        raise ValueError(f"Unrecognized NVFP4 PLE tensor: {name}")
                    loaded_buffers.add(name)
                    continue
'''+anchor)
replace('''        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):''','''        for module in self.modules():
            if isinstance(module, NVFP4PLEEmbedding):
                module.finish_load()
            if isinstance(module, Qwen3_5GatedDeltaNet):''')
replace("""            if (
                self.config.tie_word_embeddings""", """            # ModelOpt MXFP8 exports E8M0 bytes as weight_scale; SGLang's
            # block-scaled FP8 parameters use weight_scale_inv. Resolve only
            # against actual destinations, including packed projection names.
            if name.endswith(".weight_scale") and loaded_weight.dtype == torch.uint8:
                candidate = name + "_inv"
                targets = [candidate]
                for packed, unpacked, _ in stacked_params_mapping:
                    if unpacked in candidate:
                        targets.append(candidate.replace(unpacked, packed))
                targets += [x.replace("model.visual.", "visual.").replace("attn.qkv.", "attn.qkv_proj.") for x in targets[:]]
                if any(x in params_dict and params_dict[x].dtype == torch.uint8 for x in targets):
                    name = candidate
                elif name not in params_dict:
                    raise ValueError(f"No destination for MXFP8 scale: {name}")

            if (
                self.config.tie_word_embeddings""")
ast.parse(s);p.write_text(s)
print('Installed scoped TP1 NVFP4 PLE integration')
# CUTLASS MXFP8 needs N>=128; Qwen's fused GDN b/a projection has N=96.
# Its block-scale storage is already tiled/padded to 128 by FlashInfer.
p=Path('/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8_utils.py')
s=p.read_text()
anchor='''    output = flashinfer_mm_mxfp8(
        q_input,
        weight.t(),'''
assert s.count(anchor)==1
s=s.replace(anchor,'''    original_n = weight.shape[0]
    if backend == "cutlass" and original_n < 128:
        padded = torch.zeros((128, weight.shape[1]), dtype=weight.dtype, device=weight.device)
        padded[:original_n].copy_(weight)
        weight = padded

'''+anchor)
anchor='''        backend=backend,
    )

    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)'''
assert s.count(anchor)==1
s=s.replace(anchor,'''        backend=backend,
    )
    if output.shape[-1] != original_n:
        output = output[:, :original_n].contiguous()

    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)''')
ast.parse(s);p.write_text(s)
print('Installed exact small-N padding for CUTLASS MXFP8')

# Qwen's model/loader already implement language_model_only; the pinned CLI
# allowlist only contains Muse. Enable the existing Qwen implementation here.
p=Path('/sgl-workspace/sglang/python/sglang/srt/server_args.py')
s=p.read_text()
anchor='    LANGUAGE_MODEL_ONLY_ARCHITECTURES = ("MuseGlimmerForConditionalGeneration",)'
assert s.count(anchor)==1
s=s.replace(anchor,'''    LANGUAGE_MODEL_ONLY_ARCHITECTURES = (
        "MuseGlimmerForConditionalGeneration",
        "Qwen4ExpForConditionalGeneration",
        "Qwen3_8FlashNextForConditionalGeneration",
    )''')
ast.parse(s);p.write_text(s)
print('Enabled existing Qwen language-model-only implementation in CLI allowlist')

# This checkpoint uses ModelOpt's config_groups schema. The config class
# supports it, but the file reader incorrectly routes it through the older
# producer/quantization.quant_algo discriminator first.
p=Path('/sgl-workspace/sglang/python/sglang/srt/model_loader/weight_utils.py')
s=p.read_text()
anchor='''        elif model_config.quantization.startswith("modelopt") and (
            config.get("producer", {}).get("name", "").startswith("modelopt")
        ):'''
assert s.count(anchor)==1
s=s.replace(anchor,'''        elif model_config.quantization == "modelopt_mixed" and "config_groups" in config:
            return _resolve_explicit_draft_quant_config(
                model_config, quant_cls.from_config(config)
            )
'''+anchor)
ast.parse(s);p.write_text(s)
print('Routed ModelOpt mixed config_groups through the supported config parser')
