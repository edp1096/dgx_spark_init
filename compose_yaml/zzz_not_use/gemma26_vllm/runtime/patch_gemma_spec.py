"""Opt-in Gemma assistant optimizations, limited to validated TP1 paths."""
import pathlib,importlib.util
root=pathlib.Path(importlib.util.find_spec('vllm').origin).parent
p=root/'v1/attention/ops/triton_unified_attention.py';s=p.read_text()
if 'SPARKTALK_TRITON_SPEC_SPLIT' not in s:
 old='        or max_seqlen_q > 1\n        or num_seqs > seq_threshold_3D'
 new="""        or (
            max_seqlen_q > 1
            and not (
                __import__('os').environ.get('SPARKTALK_TRITON_SPEC_SPLIT') == '1'
                and max_seqlen_q <= 4
                and q.shape[0] <= softmax_segm_output.shape[0]
                and q.dtype in (torch.bfloat16, torch.float8_e4m3fn)
                and head_size in (256, 512)
                and causal
                and window_size[0] < 0
                and mm_prefix_range is None
                and rswa_prefix_lens is None
                and alibi_slopes is None
                and qq_bias is None
                and sinks is None
                and not softcap
            )
        )
        or num_seqs > seq_threshold_3D"""
 if s.count(old)!=1:raise RuntimeError('Unexpected Triton attention dispatch')
 p.write_text(s.replace(old,new))
p=root/'v1/worker/gpu_model_runner.py';s=p.read_text()
if 'SPARKTALK_GEMMA_PREFILL_SKIP' not in s:
 old="""            input_fits_in_drafter = self._input_fits_in_drafter(
                spec_decode_common_attn_metadata
            )
"""
 new=old+"""            # Gemma's Q-only drafter owns no KV state. Intermediate prefill
            # samples are discarded; the last prefill chunk still drafts normally.
            if (
                __import__('os').environ.get('SPARKTALK_GEMMA_PREFILL_SKIP') == '1'
                and spec_config.use_gemma4_mtp()
                and self.parallel_config.tensor_parallel_size == 1
                and self.parallel_config.data_parallel_size == 1
                and len(self.input_batch.req_ids) > 0
                and self.discard_request_mask.np[:len(self.input_batch.req_ids)].all()
            ):
                input_fits_in_drafter = False
"""
 if s.count(old)!=1:raise RuntimeError('Unexpected draft dispatch')
 p.write_text(s.replace(old,new))
print('Installed opt-in Gemma split-KV and intermediate-prefill skip')
