"""TP1 GDN recurrence, verify rollback/commit, prefix checkpoint and graph checks."""
from types import SimpleNamespace as NS
import torch
import pytest

from qad_gdn import recurrent, join_state_pool, verify, prefill
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating

DEVICE='cuda'
HK,HV,D=16,48,128

def inputs(tokens):
    mixed=torch.randn(tokens,(HK*2+HV)*D,device=DEVICE,dtype=torch.bfloat16)*.2
    a=torch.randn(tokens,HV,device=DEVICE,dtype=torch.bfloat16)
    b=torch.randn_like(a);z=torch.randn(tokens,HV,D,device=DEVICE,dtype=torch.bfloat16)
    alog=torch.randn(HV,device=DEVICE,dtype=torch.float32)*.1
    bias=torch.randn(HV,device=DEVICE,dtype=torch.bfloat16)*.1
    weight=torch.randn(D,device=DEVICE,dtype=torch.bfloat16)*.1+1
    return mixed,a,b,z,alog,bias,weight

def normalize(out,z,weight):
    return rms_norm_gated(x=out.reshape(-1,D),weight=weight,bias=None,z=z.reshape(-1,D),
        eps=1e-6,norm_before_gate=True,is_rms_norm=True,activation='sigmoid').view_as(z).unsqueeze(0)

def split(mixed):
    q,k,v=torch.split(mixed,[HK*D,HK*D,HV*D],dim=-1)
    return q.view(1,-1,HK,D),k.view(1,-1,HK,D),v.view(1,-1,HV,D)

def test_decode_matches_native_and_changed_graph():
    torch.manual_seed(25)
    mixed,a,b,z,alog,bias,w=inputs(2)
    state=torch.randn(5,HV,D,D,device=DEVICE)*.01;initial=state.clone();native=state.clone()
    slots=torch.tensor([3,1],device=DEVICE,dtype=torch.int64);cu=torch.tensor([0,1,2],device=DEVICE,dtype=torch.int32)
    ref_kernel=TritonGDNKernel()
    def ref():
        out=ref_kernel.packed_decode(mixed,a,b,A_log=alog,dt_bias=bias,scale=D**-.5,
            ssm_states=native,cache_indices=slots,num_v_heads=HV,head_v_dim=D)
        return normalize(out,z,w)
    def run():return recurrent(mixed,a,b,z,w,alog,bias,state,slots[:,None],cu)
    actual=run();expected=ref()
    torch.testing.assert_close(actual,expected,rtol=.025,atol=.008)
    torch.testing.assert_close(state,native,rtol=3e-4,atol=3e-5)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):state.copy_(initial);run()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):captured=run()
    mixed.copy_(torch.randn_like(mixed)*.2);z.copy_(torch.randn_like(z))
    state.copy_(initial);native.copy_(initial)
    torch.cuda.synchronize();graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(captured,ref(),rtol=.025,atol=.008)
    torch.testing.assert_close(state,native,rtol=3e-4,atol=3e-5)

def test_verify_keeps_persistent_state_and_exports_each_accept_prefix():
    torch.manual_seed(26);steps=4;batch=2
    mixed,a,b,z,alog,bias,w=inputs(batch*steps)
    persistent=torch.randn(1,5,HV,D,D,device=DEVICE)*.01
    inter=torch.zeros(1,3,steps,HV,D,D,device=DEVICE)
    pool=NS(mamba_cache=MambaPool.SpeculativeState(conv=[],temporal=persistent,
        intermediate_ssm=inter,intermediate_conv_window=[]))
    join_state_pool(pool)
    assert pool.mamba_cache.temporal.untyped_storage().data_ptr()==pool.qad_state_pool.untyped_storage().data_ptr()
    assert pool.mamba_cache.intermediate_ssm.untyped_storage().data_ptr()==pool.qad_state_pool.untyped_storage().data_ptr()
    initial=pool.mamba_cache.temporal[0].clone();native=initial.clone();native_inter=torch.zeros_like(inter[0])
    slots=torch.tensor([3,1],device=DEVICE,dtype=torch.int64)
    rows=torch.tensor([2,0],device=DEVICE,dtype=torch.int32)
    cu=torch.tensor([0,steps,2*steps],device=DEVICE,dtype=torch.int32)
    backend=NS(req_to_token_pool=NS(mamba_pool=pool,mamba2_layer_index=lambda _:0),
        forward_metadata=NS(mamba_cache_indices=slots,query_start_loc=cu),verify_intermediate_state_indices=rows)
    layer=NS(layer_id=0,A_log=alog,dt_bias=bias)
    fb=NS(spec_info=NS(topk=1,draft_token_num=steps))
    oracle_state=pool.qad_state_pool[0].clone()
    from b12x.sequence.gdn_decode.reference import decode as reference_decode
    oracle_output=reference_decode(mixed,a,b,z,alog,bias,w,oracle_state,cu,
        torch.tensor([5,5],device=DEVICE,dtype=torch.int32),
        torch.tensor([[13,14,15,16,3],[5,6,7,8,1]],device=DEVICE,dtype=torch.int64),
        2,8,key_heads=HK,value_heads=HV,gate_activation='sigmoid',null_state_index=-1)
    actual=verify(backend,layer,fb,mixed,a,b,dict(qad_z=z,qad_norm_weight=w,qad_gate='sigmoid',qad_eps=1e-6))
    q,k,v=split(mixed)
    raw=TritonGDNKernel().target_verify(alog,bias,q,k,v,a,b,ssm_states=native,
        cache_indices=slots,query_start_loc=cu,intermediate_states_buffer=native_inter,
        intermediate_state_indices=rows,cache_steps=steps,retrieve_parent_token=None)
    torch.testing.assert_close(actual,normalize(raw,z,w),rtol=.025,atol=.008)
    torch.testing.assert_close(pool.mamba_cache.temporal[0],initial,rtol=0,atol=0)
    # B12X's public oracle rounds beta to BF16; SGLang's target-verify
    # recurrence keeps beta in FP32. Validate the B12X contract tightly,
    # and separately bound/report the expected change against the old path.
    torch.testing.assert_close(actual.squeeze(0),oracle_output,rtol=.015,atol=.004)
    torch.testing.assert_close(pool.qad_state_pool[0],oracle_state,rtol=3e-4,atol=3e-5)
    error=(pool.mamba_cache.intermediate_ssm[0]-native_inter).norm()/native_inter.norm()
    print('GDN verify state relative L2 versus FP32-beta SGLang:',float(error))
    assert error<.004
    # Accept one token from request 0 and all four from request 1; rejected
    # checkpoints must never leak into the persistent state.
    accepted=torch.tensor([0,3],device=DEVICE)
    pool.mamba_cache.temporal[0,slots]=pool.mamba_cache.intermediate_ssm[0,rows.long(),accepted]
    expected_commit=initial.clone()
    expected_commit[slots]=oracle_state[torch.tensor([13,8],device=DEVICE)]
    torch.testing.assert_close(pool.mamba_cache.temporal[0],expected_commit,rtol=3e-4,atol=3e-5)

def test_prefill_matches_native_final_and_prefix_checkpoint(monkeypatch):
    import sglang.srt.runtime_context as context
    monkeypatch.setattr(context,'mamba_cache_chunk_size',lambda:64)
    torch.manual_seed(27);tokens=97
    mixed,a,b,z,alog,bias,w=inputs(tokens)
    state=torch.randn(6,HV,D,D,device=DEVICE)*.01;native=state.clone()
    slots=torch.tensor([2],device=DEVICE,dtype=torch.int64)
    cu=torch.tensor([0,tokens],device=DEVICE,dtype=torch.int32)
    backend=NS(forward_metadata=NS(query_start_loc=cu,has_mamba_track_mask=True))
    layer=NS(num_k_heads=HK,num_v_heads=HV,q_dim=HK*D,k_dim=HK*D,v_dim=HV*D,A_log=alog,dt_bias=bias)
    fb=NS(extend_prefix_lens=torch.tensor([100],device=DEVICE),
        mamba_track_mask=torch.tensor([True],device=DEVICE),
        mamba_track_indices=torch.tensor([4],device=DEVICE),
        mamba_track_seqlens=torch.tensor([196],device=DEVICE))
    actual=prefill(backend,layer,fb,mixed,a,b,state,slots)
    q,k,v=split(mixed);g,beta=fused_gdn_gating(alog,a,b,bias)
    expected,_,h=TritonGDNKernel().extend(q,k,v,g,beta,ssm_states=native,cache_indices=slots,query_start_loc=cu)
    torch.testing.assert_close(actual,expected,rtol=.04,atol=.008)
    torch.testing.assert_close(state[2],native[2],rtol=.008,atol=.003)
    torch.testing.assert_close(state[4],h.squeeze(0)[1].float(),rtol=.008,atol=.003)


def test_joined_multilayer_pool_uses_native_commit_and_track_scatter():
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import scatter_mamba_states_after_mtp_verify
    torch.manual_seed(28)
    original=torch.randn(2,5,HV,D,D,device=DEVICE)
    intermediate=torch.randn(2,3,4,HV,D,D,device=DEVICE)
    pool=NS(mamba_cache=MambaPool.SpeculativeState(conv=[],temporal=original,
        intermediate_ssm=intermediate,intermediate_conv_window=[]))
    join_state_pool(pool)
    # Joined per-layer allocations deliberately give the all-layer temporal
    # and intermediate views non-contiguous outer strides.
    assert not pool.mamba_cache.temporal.is_contiguous()
    slots=torch.tensor([3,1],device=DEVICE,dtype=torch.int64)
    accept=torch.tensor([0,3],device=DEVICE,dtype=torch.int64)
    track=torch.tensor([4,-1],device=DEVICE,dtype=torch.int64)
    track_steps=torch.tensor([2,-1],device=DEVICE,dtype=torch.int64)
    expected=original.clone()
    expected[:,3]=intermediate[:,0,0]
    expected[:,1]=intermediate[:,1,3]
    expected[:,4]=intermediate[:,0,2]
    scatter_mamba_states_after_mtp_verify(pool.mamba_cache,slots,accept,track,track_steps)
    torch.testing.assert_close(pool.mamba_cache.temporal,expected,rtol=0,atol=0)
    torch.testing.assert_close(pool.mamba_cache.intermediate_ssm,intermediate,rtol=0,atol=0)


@pytest.mark.parametrize('physical,live,tracked',[(128,97,4),(97,97,4),(128,128,2)])
def test_prefill_padding_aligned_tail_and_self_track(monkeypatch,physical,live,tracked):
    import sglang.srt.runtime_context as context
    monkeypatch.setattr(context,'mamba_cache_chunk_size',lambda:64)
    torch.manual_seed(32)
    mixed,a,b,z,alog,bias,w=inputs(physical)
    state=torch.randn(6,HV,D,D,device=DEVICE)*.01;native=state.clone()
    slots=torch.tensor([2],device=DEVICE,dtype=torch.int64)
    cu=torch.tensor([0,live],device=DEVICE,dtype=torch.int32)
    backend=NS(forward_metadata=NS(query_start_loc=cu,has_mamba_track_mask=True))
    layer=NS(num_k_heads=HK,num_v_heads=HV,q_dim=HK*D,k_dim=HK*D,v_dim=HV*D,A_log=alog,dt_bias=bias)
    # Native aligned tracking copies the final state, even if the live tail
    # is not aligned to B12X's 16-token checkpoint export boundary.
    fb=NS(extend_prefix_lens=torch.tensor([100],device=DEVICE),
        mamba_track_mask=torch.tensor([True],device=DEVICE),
        mamba_track_indices=torch.tensor([tracked],device=DEVICE),
        mamba_track_seqlens=torch.tensor([164],device=DEVICE))
    actual=prefill(backend,layer,fb,mixed,a,b,state,slots)
    q,k,v=split(mixed[:live]);g,beta=fused_gdn_gating(alog,a[:live],b[:live],bias)
    expected,_,_=TritonGDNKernel().extend(q,k,v,g,beta,ssm_states=native,cache_indices=slots,query_start_loc=cu)
    torch.testing.assert_close(actual[:,:live],expected,rtol=.04,atol=.008)
    torch.testing.assert_close(state[2],native[2],rtol=.008,atol=.003)
    torch.testing.assert_close(state[tracked],state[2],rtol=0,atol=0)
    assert torch.count_nonzero(actual[:,live:])==0
