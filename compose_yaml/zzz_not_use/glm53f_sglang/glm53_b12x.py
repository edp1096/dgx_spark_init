"""GLM5Next sparse attention bridge; uses B12X's native 528-byte FP8 recipe."""
import torch
from b12x.attention import sparse_mla


class SparseAttention:
    def __init__(self):
        self.signature = None
        self.plan = self.storage = self.lengths = None
        # Graphs capture scratch addresses, including multi-token draft verification.
        # Keep captured plans alive when a later eager prefill changes shape.
        self.graph_plans = {}

    def __call__(self, query, cache, indices, *, scale, page_size=64):
        if query.dtype != torch.bfloat16 or query.ndim != 3 or query.shape[-1] != 512:
            raise ValueError('GLM5Next B12X requires [tokens, heads, 512] BF16 queries')
        if cache.shape[-1] != 528 or cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise ValueError('GLM5Next B12X requires the native 528-byte FP8 cache')
        if indices.ndim != 2 or indices.shape[0] != query.shape[0] or indices.dtype != torch.int32:
            raise ValueError('Sparse indices must be [tokens, selected slots] int32')
        if torch.cuda.get_device_capability(query.device) not in ((12, 0), (12, 1)):
            raise ValueError('This bridge is qualified only for SM120/SM121')
        if not sparse_mla.is_supported(query.device):
            raise RuntimeError('Native B12X kernel unavailable; reference fallback is disallowed')
        query = query.contiguous()
        indices = indices.contiguous()
        cache = cache.view(torch.uint8)
        if cache.ndim != 3:
            raise ValueError('Cache must be rank 3')
        rows, heads, _ = query.shape
        width = indices.shape[1]
        signature = (rows, heads, width, float(scale), page_size, query.device,
                     torch.cuda.current_stream(query.device).cuda_stream)
        capturing = torch.cuda.is_current_stream_capturing()
        cached = self.graph_plans.get(signature)
        if cached is not None:
            plan, storage, lengths = cached
        elif signature != self.signature:
            if capturing:
                raise RuntimeError('B12X plan must be warmed before graph capture')
            caps = sparse_mla.Caps(device=query.device, num_q_heads=heads,
                max_q_rows=rows, max_width=width, softmax_scale=float(scale),
                dtype=query.dtype, kv_dtype=torch.uint8, head_dim=512, v_head_dim=512,
                model_type=sparse_mla.ModelType.GLM_NEXT, cache_record_bytes=528,
                mode='decode' if rows == 1 else 'extend', page_size=page_size)
            self.plan = sparse_mla.plan(caps)
            self.storage = [torch.empty(shape, dtype=dtype, device=query.device)
                            for shape, dtype in self.plan.shapes_and_dtypes()]
            # KPOOL tails follow padding holes; count_nonzero would truncate them.
            # Every column is traversed; -1 entries are masked by the kernel.
            self.lengths = torch.full((rows,), width, dtype=torch.int32, device=query.device)
            self.signature = signature
            plan, storage, lengths = self.plan, self.storage, self.lengths
        else:
            plan, storage, lengths = self.plan, self.storage, self.lengths
        if capturing:
            self.graph_plans[signature] = (plan, storage, lengths)
        binding = sparse_mla.bind(plan, scratch=storage, q=query,
            kv_cache=cache, selected_indices=indices, cache_lengths=lengths,
            selected_lengths=lengths)
        return sparse_mla.run(binding)
