"""Compare QAD B12X snapshot verify with native SGLang ReplaySSM on TP1 geometry.
No model weights or serving changes. Runs a single GDN layer, then scales bytes
(not measured latency) to the model's configured GDN layer count.
"""
import json
import torch
import triton.testing
from qad_gdn import recurrent
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import commit_gdn_replayssm_fold_all_layers
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated

def bench(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream): fn()
    for _ in range(10): graph.replay()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(200):graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1000/200

def case(steps):
    torch.manual_seed(731)
    hk,hv,d,slots=16,48,128,3
    rand=lambda *shape: torch.randn(*shape,device='cuda',dtype=torch.bfloat16)
    mixed=rand(steps,(hk*2+hv)*d)*.2
    a,b,z=rand(steps,hv),rand(steps,hv),rand(steps,hv,d)
    alog=torch.randn(hv,device='cuda')*.1; bias=rand(hv)*.1; w=rand(d)*.1+1
    state=torch.randn(slots+steps,hv,d,d,device='cuda')*.01
    original=state.clone(); native=state[:slots].clone()
    ids=torch.tensor([1],device='cuda',dtype=torch.int64)
    cu=torch.tensor([0,steps],device='cuda',dtype=torch.int32)
    indices=torch.tensor([[*range(slots,slots+steps),1]],device='cuda',dtype=torch.int64)
    rawv=torch.zeros(1,slots,hv,steps,d,device='cuda',dtype=torch.bfloat16)
    rawk=torch.zeros(1,slots,hk,steps,d,device='cuda',dtype=torch.bfloat16)
    gates=torch.zeros(1,slots,hv,steps,device='cuda'); beta=torch.zeros_like(gates)
    accept=torch.tensor([steps],device='cuda',dtype=torch.int32)
    q,k,v=torch.split(mixed,[hk*d,hk*d,hv*d],dim=-1)
    q=q.reshape(1,steps,hk,d); k=k.reshape_as(q); v=v.reshape(1,steps,hv,d)
    def snapshot():return recurrent(mixed,a,b,z,w,alog,bias,state,indices,cu)
    def ring():
        out=fused_sigmoid_gating_delta_rule_update(A_log=alog,dt_bias=bias,q=q,k=k,v=v,a=a,b=b,
            initial_state_source=native,initial_state_indices=ids,cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,softplus_beta=1.,softplus_threshold=20.,is_kda=False,
            disable_state_update=True,cache_ring=True,replayssm_rawv=rawv[0],replayssm_rawk=rawk[0],
            replayssm_g=gates[0],replayssm_beta=beta[0])
        return rms_norm_gated(x=out.reshape(-1,d),weight=w,bias=None,z=z.reshape(-1,d),
            eps=1e-6,norm_before_gate=True,is_rms_norm=True,activation='sigmoid').reshape_as(z)
    def fold():
        commit_gdn_replayssm_fold_all_layers(checkpoint_state=native.unsqueeze(0),rawv_cache=rawv,
            rawk_cache=rawk,g_cache=gates,beta_cache=beta,ssm_state_indices=ids,accept_lens=accept,
            max_cache_len=steps,num_k_heads=hk,null_block_id=-1)
    out=snapshot().reshape_as(z); replay_out=ring()
    torch.testing.assert_close(out,replay_out,rtol=.025,atol=.008)
    errors=[]
    for n in range(1,steps+1):
        native.copy_(original[:slots]); ring();accept.fill_(n);fold();torch.cuda.synchronize()
        expected=state[slots+n-1]
        errors.append(float((native[1]-expected).abs().max()))
        torch.testing.assert_close(native[1],expected,rtol=.005,atol=.0005)
        torch.testing.assert_close(native[[0,2]],original[[0,2]],rtol=0,atol=0)
    native.copy_(original[:slots]);accept.fill_(steps)
    # Restore the initial checkpoint each iteration for a stable workload;
    # this reset is included equally in paired end-to-end timings below.
    def baseline_cycle():
        state[1].copy_(original[1]);snapshot();state[1].copy_(state[slots+steps-1])
    def replay_cycle():
        native[1].copy_(original[1]);ring();fold()
    result=dict(steps=steps,b12x_verify_us=bench(snapshot),native_ring_verify_us=bench(ring),
        native_fold_us=bench(fold),snapshot_cycle_us=bench(baseline_cycle),replay_cycle_us=bench(replay_cycle),
        snapshot_bytes_per_layer=steps*hv*d*d*4,
        ring_bytes_per_live_slot_layer=(hv+hk)*steps*d*2+hv*steps*8,
        max_state_error_by_accepted_length=errors)
    print(json.dumps(result),flush=True)
    return result
if __name__=='__main__':
    print(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}),flush=True)
    for steps in (2,4,6):case(steps)
