"""Register a narrow GLM5Next B12X backend on a pinned SGLang source."""
import ast
from pathlib import Path
import sglang
root=Path(sglang.__file__).parent

def patch(path,old,new,count=1):
    p=root/path;s=p.read_text()
    if s.count(old)!=count:raise RuntimeError(f'{path}: expected {count} source anchors, found {s.count(old)}')
    s=s.replace(old,new);ast.parse(s);p.write_text(s)

patch('srt/arg_groups/fields/exec_.py','                "flashinfer_sparse_mla",','                "flashinfer_sparse_mla",\n                "b12x_glm",',2)
patch('srt/layers/attention/dsa_backend.py','    "flashinfer_sparse_mla",','    "flashinfer_sparse_mla",\n    "b12x_glm",')
anchor='        uses_flashinfer_sparse_mla = _validate_flashinfer_sparse_mla_backend('
patch('srt/layers/attention/dsa_backend.py',anchor,'''        if "b12x_glm" in (self.dsa_prefill_impl, self.dsa_decode_impl):
            if (model_runner.model_config.hf_config.architectures[0] != "Glm5NextForConditionalGeneration"
                    or self.device_sm_major != 12
                    or self.kv_cache_dtype != torch.float8_e4m3fn
                    or self.kv_lora_rank != 512 or self.qk_rope_head_dim != 0
                    or self.kv_cache_dim != 528):
                raise ValueError("b12x_glm requires GLM5Next SM12x and native 528-byte FP8 KV")
            from glm53_b12x import SparseAttention
            self._glm53_sparse = SparseAttention()

'''+anchor)
patch('srt/layers/attention/dsa_backend.py','        elif dsa_impl == "flashinfer_sparse_mla":','        elif dsa_impl in ("flashinfer_sparse_mla", "b12x_glm"):',2)
patch('srt/layers/attention/dsa_backend.py','            return self._forward_flashinfer_sparse_mla(','            return (self._forward_b12x_glm if dsa_impl == "b12x_glm" else self._forward_flashinfer_sparse_mla)(',2)
anchor='    def _forward_flashinfer_sparse_mla('
patch('srt/layers/attention/dsa_backend.py',anchor,'''    def _forward_b12x_glm(self, q_all, kv_cache, page_table_1, seq_lens,
                           sm_scale, skip_softmax_threshold_scale_factor):
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError("b12x_glm does not implement approximate softmax skipping")
        return self._glm53_sparse(q_all, kv_cache, page_table_1,
                                 scale=sm_scale, page_size=self.real_page_size)

'''+anchor)
patch('srt/layers/attention/dsa/dsa_backend_kpool.py','dsa_impl in ("fa3", "tilelang", "trtllm")','dsa_impl in ("fa3", "tilelang", "trtllm", "b12x_glm")')
print('GLM5Next B12X backend registered',root)
