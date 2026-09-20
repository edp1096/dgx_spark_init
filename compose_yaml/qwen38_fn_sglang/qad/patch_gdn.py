"""Install opt-in GDN hooks against the pinned SGLang source, failing on drift."""
from pathlib import Path
import ast

ROOT=Path('/sgl-workspace/sglang/python/sglang/srt')
def patch(path,old,new):
    p=ROOT/path;s=p.read_text()
    if s.count(old)!=1:raise RuntimeError(f'Unexpected source anchor: {path}: {old[:90]}')
    s=s.replace(old,new);ast.parse(s);p.write_text(s)

patch('mem_cache/memory_pool.py',
      '            mem_usage_bytes = self.mamba_cache.mem_usage_bytes()\n',
      '''            if os.environ.get("SGLANG_QAD_B12X_GDN") == "1":
                from qad_gdn import join_state_pool
                join_state_pool(self)
            mem_usage_bytes = self.mamba_cache.mem_usage_bytes()
''')
patch('../kernels/ops/mamba/mamba_state_scatter_triton.py',
'''    if not src.is_contiguous():
        raise ValueError("src tensor must be contiguous")
''',
'''    # The kernel already receives layer/request/step strides explicitly.
    # Joined GDN pools pad only the layer stride; each state entry remains
    # contiguous. Keep rejecting non-contiguous state entries themselves.
    _require_entry_contiguous_dst(src, 3, "fused_mamba_state_scatter_with_mask src")
''')
patch('models/qwen3_5.py','import torch\n','import torch\nfrom qad_gdn import ENABLED as QAD_B12X_GDN\n')
patch('models/qwen3_5.py',
'''        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        )
''',
'''        qad_fused_norm = False
        if QAD_B12X_GDN:
            from sglang.srt.model_executor.forward_context import get_attn_backend
            if self.attn_tp_size != 1:
                raise ValueError("QAD B12X GDN requires TP1")
            if forward_batch.forward_mode.is_draft_extend_v2():
                raise ValueError("QAD GDN draft-extend is not qualified; Qwen4 MTP uses QSA")
            qad_fused_norm = (forward_batch.forward_mode.is_decode()
                              or forward_batch.forward_mode.is_target_verify())
            core_attn_out = get_attn_backend().forward(
                layer=self.attn, forward_batch=forward_batch, mixed_qkv=mixed_qkv,
                a=a, b=b, qad_z=z, qad_norm_weight=self.norm.weight,
                qad_gate=self.output_gate_type or "silu", qad_eps=self.layer_norm_epsilon,
            )
        else:
            core_attn_out = self.attn(
                forward_batch, mixed_qkv=mixed_qkv, a=a, b=b,
            )
''')
patch('models/qwen3_5.py','        core_attn_out = self.norm(core_attn_out, z)\n',
      '        if not qad_fused_norm:\n            core_attn_out = self.norm(core_attn_out, z)\n')
patch('layers/attention/linear/gdn_backend.py','import torch\n',
      'import torch\nfrom qad_gdn import ENABLED as QAD_B12X_GDN\n')
patch('layers/attention/linear/gdn_backend.py',
'''        # Skip split + reshape + separate gating kernel by consuming
''',
'''        if QAD_B12X_GDN:
            from qad_gdn import decode
            output = decode(self, layer, forward_batch, mixed_qkv, a, b, kwargs)
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
            )
            return output

        # Skip split + reshape + separate gating kernel by consuming
''')
patch('layers/attention/linear/gdn_backend.py',
'''        actual_seq_len = mixed_qkv.shape[0]
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
''',
'''        if QAD_B12X_GDN:
            from qad_gdn import verify, prefill
            if is_target_verify:
                return verify(self, layer, forward_batch, mixed_qkv, a, b, kwargs)
            if needs_state_gather:
                raise ValueError("QAD B12X GDN requires contiguous per-layer state slots")
            return prefill(self, layer, forward_batch, mixed_qkv, a, b,
                           ssm_states_contig, state_cache_indices)

        actual_seq_len = mixed_qkv.shape[0]
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
''')
print('Installed opt-in QAD B12X GDN hooks')
