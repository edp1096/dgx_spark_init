"""TP2 MoE with SSD-backed original weights and native b12x expert graphs.

The full model is eager; the b12x backend supports DSpark draft layers.
Marlin and decoded-BF16 implementations remain diagnostic fallbacks.
"""
import os
import re
import weakref
import torch
import torch.nn.functional as F
from expert_store import ExpertStore
import step_profile
import expert_io
from streaming_graphs import expert_io_break

_store = None

@expert_io_break
def streamed_experts_with_output(x,weights,ids,output,layer,limit):
    """The CPU routing read/slot update is replayed between GPU graph segments."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    if BreakableCUDAGraphCapture.current() is not None:
        # Captured prefix kernels have not executed yet: their route IDs may
        # be uninitialized. Warmup already runs the real experts; capture only
        # needs a finite placeholder in the fixed output buffer. Replay calls
        # this same function with no active capture and uses the real routes.
        output.zero_()
        return
    import b12x_slots
    store=get_store()
    output.copy_(b12x_slots.apply(store,layer,x,weights,ids,limit))
    if layer==39:
        import json
        print('EXPERT_STREAM '+json.dumps(store.stats|{'rank':store.rank,
              'cache_bytes':store.resident_bytes}|b12x_slots.stats()),flush=True)

def get_store():
    global _store
    if _store is None:
        from vllm.distributed import get_tensor_model_parallel_rank
        _store = ExpertStore(os.environ['DSV41_MODEL'],
                             rank=get_tensor_model_parallel_rank(),
                             cache_bytes=(0 if os.environ.get('DSV41_MOE_BACKEND') in ('marlin','b12x_slots') else int(float(os.environ.get('DSV41_EXPERT_CACHE_GIB', '8'))*2**30)))
    return _store


def dequant(raw_weight, raw_scale, device):
    raw, shape = raw_weight
    weight = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape).to(device)
    raw, shape = raw_scale
    scale = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape).to(device)
    # E2M1 packed low nibble then high nibble; E8M0 unsigned exponent.
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=device)
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2)
    factors = torch.exp2(scale.float()-127)
    factors = factors.masked_fill(scale == 255, float('nan'))
    return (lut[codes.long()].reshape(*scale.shape, 32)*factors.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


class StreamingExperts(torch.nn.Module):
    def __init__(self, owner, prefix):
        super().__init__()
        # Do not register gate/shared modules twice: their canonical loader
        # names must remain ffn.gate and ffn.shared_experts.
        self.owner_ref = weakref.ref(owner)
        match = re.search(r'layers\.(\d+)\.ffn$', prefix)
        if not match or int(match[1]) >= 43:
            raise ValueError(f'Unsupported target/draft layer prefix: {prefix}')
        self.layer_index = int(match[1])
        self.capture_fn = None
        self.prefetch_enabled = os.environ.get('DSV41_PREFETCH_TEST') == '1'

    def forward(self, hidden_states, router_logits=None, input_ids=None):
        from vllm.distributed import tensor_model_parallel_all_reduce
        from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
        owner = self.owner_ref()
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        step_profile.begin(self.layer_index, x.shape[0])
        with step_profile.phase('router'):
            logits, _ = owner.gate(x)
            bias_vl = getattr(owner.gate, 'bias_vl', None)
            weights, ids = fused_topk_bias(
                hidden_states=x, gating_output=logits,
                scoring_func=owner.scoring_func,
                e_score_correction_bias=owner.gate.e_score_correction_bias,
                topk=owner.n_activated_experts, renormalize=owner.renormalize,
                indices_type=torch.int32, input_tokens=input_ids,
                hash_indices_table=None, routed_scaling_factor=owner.routed_scaling_factor,
                bias_vl=bias_vl, image_sentinel_lo=owner.image_sentinel_lo if bias_vl is not None else 0)
        store = get_store()
        selected = getattr(self, "output_rows", None)
        shared_result=None
        def start_shared():
            nonlocal shared_result
            if shared_result is not None: return False
            shared_result=owner.shared_experts(x).float()
            return True
        overlap=(expert_io.overlaps() and owner.shared_experts is not None)
        if os.environ.get('DSV41_MOE_BACKEND') == 'b12x_slots':
            import b12x_slots
            if os.environ.get('DSV41_MODEL_GRAPHS')=='1':
                out=torch.empty_like(x,dtype=torch.float32)
                streamed_experts_with_output(x,weights,ids,out,self.layer_index,owner.swiglu_limit)
            else:
                if selected is None:
                    out = b12x_slots.apply(store, self.layer_index, x, weights, ids, owner.swiglu_limit,
                                          on_load_start=start_shared if overlap else None)
                else:
                    assert self.layer_index == 39
                    partial = b12x_slots.apply(store, self.layer_index,
                        x.index_select(0, selected), weights.index_select(0, selected),
                        ids.index_select(0, selected), owner.swiglu_limit,
                        on_load_start=start_shared if overlap else None)
                    out = torch.zeros_like(x, dtype=torch.float32)
                    out.index_copy_(0, selected, partial)
        elif os.environ.get('DSV41_MOE_BACKEND') == 'marlin':
            import marlin_stream
            out = marlin_stream.apply(store, self.layer_index, x, weights, ids, owner.swiglu_limit).float()
        else:
            unique = ids.unique().tolist()  # explicit eager sync; disallow graph capture
            out = torch.zeros_like(x, dtype=torch.float32)
            for expert in unique:
                token, choice = torch.where(ids == expert)
                chunk = x[token]
                tensors = store.expert(self.layer_index, expert)
                gate = F.linear(chunk, dequant(tensors['w1.weight'], tensors['w1.scale'], x.device)).float()
                up = F.linear(chunk, dequant(tensors['w3.weight'], tensors['w3.scale'], x.device)).float()
                if owner.swiglu_limit is not None and owner.swiglu_limit > 0:
                    gate = gate.clamp(max=owner.swiglu_limit)
                    up = up.clamp(-owner.swiglu_limit, owner.swiglu_limit)
                intermediate = F.silu(gate)*up
                # Follow DeepSeek's reference expert: router weight before w2.
                intermediate = (intermediate*weights[token, choice, None]).to(x.dtype)
                result = F.linear(intermediate, dequant(tensors['w2.weight'], tensors['w2.scale'], x.device))
                out.index_add_(0, token, result.float())
        with step_profile.phase("shared"):
            if owner.shared_experts is not None:
                out = out + (shared_result if shared_result is not None else owner.shared_experts(x).float())
        with step_profile.phase("tp_reduce"):
            if os.environ.get("DSV41_MOE_BACKEND") == "b12x_slots":
                out = tensor_model_parallel_all_reduce(out).to(x.dtype)
            else:
                out = tensor_model_parallel_all_reduce(out.to(x.dtype))
        step_profile.end(self.layer_index)
        if self.layer_index == 39 and os.environ.get('DSV41_MODEL_GRAPHS')!='1':
            import json
            extra = (b12x_slots.stats() if os.environ.get("DSV41_MOE_BACKEND") == "b12x_slots" else
                     marlin_stream.stats() if os.environ.get("DSV41_MOE_BACKEND") == "marlin" else {})
            print("EXPERT_STREAM " + json.dumps(store.stats | {"rank": store.rank, "cache_bytes": store.resident_bytes} | extra), flush=True)
        return out.reshape(hidden_states.shape)


def install(DeepseekV4MoE):
    def init_streaming(self, vllm_config, config, quant_config, prefix):
        graphs=os.environ.get('DSV41_MODEL_GRAPHS')=='1'
        if graphs:
            from vllm.config import CUDAGraphMode
            if (os.environ.get('DSV41_MOE_BACKEND')!='b12x_slots'
                    or os.environ.get('DSV41_PREFETCH_TEST')=='1'
                    or vllm_config.parallel_config.data_parallel_size!=1
                    or vllm_config.compilation_config.cudagraph_mode not in
                       (CUDAGraphMode.PIECEWISE,CUDAGraphMode.NONE,CUDAGraphMode.FULL,
                        CUDAGraphMode.FULL_DECODE_ONLY)):
                raise RuntimeError('Streaming model graphs require supported b12x TP2/DP1 capture without forecast prefetch')
            if (vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
                    and os.environ.get('DSV41_ATTENTION_GRAPHS')!='1'):
                raise RuntimeError('FULL streaming graphs require the breakable attention graph manager')
        if (self.tp_size != 2 or self.use_sequence_parallel
                or vllm_config.parallel_config.enable_expert_parallel
                or vllm_config.parallel_config.enable_eplb
                or (vllm_config.speculative_config is not None and os.environ.get("DSV41_MOE_BACKEND") != "b12x_slots")
                or (not vllm_config.model_config.enforce_eager and not graphs)):
            raise RuntimeError('Streaming reference requires TP2, eager, no EP/EPLB/speculation')
        self.n_redundant_experts = 0
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_logical_experts = self.n_physical_experts = self.n_routed_experts
        self.n_local_physical_experts = self.n_local_experts = self.n_routed_experts // 2
        self.experts = StreamingExperts(self, prefix)
    DeepseekV4MoE._init_fused_moe_experts = init_streaming
