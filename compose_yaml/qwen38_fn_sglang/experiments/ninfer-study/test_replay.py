"""Native ReplaySSM identity and QAD adapter opt-in/commit regression."""
from types import SimpleNamespace as NS
import os
import torch
from qad_gdn import join_state_pool,verify
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import commit_gdn_replayssm_fold_all_layers
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update

def test_closed_loop():
    torch.manual_seed(732)
    steps,slots,hk,hv,d=4,4,16,48,128
    initial=torch.randn(1,slots,hv,d,d,device='cuda')*.01
    temporal=initial.clone();reference=initial.clone()
    rv=torch.zeros(1,slots,hv,steps,d,device='cuda',dtype=torch.bfloat16)
    rk=torch.zeros(1,slots,hk,steps,d,device='cuda',dtype=torch.bfloat16)
    g=torch.zeros(1,slots,hv,steps,device='cuda');beta=torch.zeros_like(g)
    state=NS(temporal=temporal,intermediate_ssm=None,replayssm_rawv=rv,replayssm_rawk=rk,
        replayssm_d=None,replayssm_g=g,replayssm_beta=beta)
    pool=NS(mamba_cache=state,replayssm_spec_fold=True,replayssm_is_kda=False)
    os.environ.pop('SGLANG_QAD_REPLAY_VERIFY',None)
    try:join_state_pool(pool)
    except ValueError:pass
    else:raise AssertionError('missing opt-in accepted')
    os.environ['SGLANG_QAD_REPLAY_VERIFY']='1';join_state_pool(pool)
    ids=torch.tensor([1],device='cuda',dtype=torch.int64);cu=torch.tensor([0,steps],device='cuda',dtype=torch.int32)
    lc=NS(temporal=temporal[0],replayssm_rawv=rv[0],replayssm_rawk=rk[0],replayssm_g=g[0],replayssm_beta=beta[0])
    backend=NS(req_to_token_pool=NS(mamba_pool=pool,mamba2_layer_cache=lambda _:lc),
        forward_metadata=NS(mamba_cache_indices=ids,query_start_loc=cu),
        kernel_dispatcher=NS(verify_kernel_is_flashinfer=False))
    backend._replayssm_fold_target_verify=lambda **kw:GDNAttnBackend._replayssm_fold_target_verify(backend,**kw)
    alog=torch.randn(hv,device='cuda')*.1;bias=torch.randn(hv,device='cuda',dtype=torch.bfloat16)*.1
    layer=NS(layer_id=0,q_dim=hk*d,k_dim=hk*d,v_dim=hv*d,A_log=alog,dt_bias=bias)
    weight=torch.ones(d,device='cuda',dtype=torch.bfloat16)
    scratch=torch.zeros(1,steps,hv,d,d,device='cuda')
    row=torch.zeros(1,device='cuda',dtype=torch.int32)
    accept=torch.ones(1,device='cuda',dtype=torch.int32)
    track=torch.tensor([2],device='cuda',dtype=torch.int64);trackstep=torch.tensor([0],device='cuda',dtype=torch.int64)
    for turn in range(32):
        mixed=torch.randn(steps,(hk*2+hv)*d,device='cuda',dtype=torch.bfloat16)*.2
        a=torch.randn(steps,hv,device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a)
        z=torch.randn(steps,hv,d,device='cuda',dtype=torch.bfloat16)
        before=temporal.clone()
        output=verify(backend,layer,NS(spec_info=NS(topk=1,draft_token_num=steps)),mixed,a,b,
            dict(qad_z=z,qad_norm_weight=weight,qad_gate='sigmoid',qad_eps=1e-6))
        assert torch.isfinite(output).all()
        torch.testing.assert_close(temporal,before,rtol=0,atol=0)
        q,k,v=torch.split(mixed,[hk*d,hk*d,hv*d],-1)
        raw=fused_sigmoid_gating_delta_rule_update(A_log=alog,dt_bias=bias,
            q=q.reshape(1,steps,hk,d),k=k.reshape(1,steps,hk,d),v=v.reshape(1,steps,hv,d),a=a,b=b,
            initial_state_source=reference[0],initial_state_indices=ids,cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,softplus_beta=1.,softplus_threshold=20.,disable_state_update=True,
            intermediate_states_buffer=scratch,intermediate_state_indices=row,cache_steps=steps)
        expected=rms_norm_gated(x=raw.reshape(-1,d),weight=weight,bias=None,z=z.reshape(-1,d),
            eps=1e-6,norm_before_gate=True,is_rms_norm=True,activation='sigmoid').reshape_as(output)
        torch.testing.assert_close(output,expected,rtol=0,atol=0)
        n=turn%steps+1;accept.fill_(n)
        commit_gdn_replayssm_fold_all_layers(checkpoint_state=temporal,rawv_cache=rv,rawk_cache=rk,
            g_cache=g,beta_cache=beta,ssm_state_indices=ids,accept_lens=accept,max_cache_len=steps,
            num_k_heads=hk,mamba_track_indices=track,mamba_steps_to_track=trackstep,null_block_id=-1)
        reference[0,1].copy_(scratch[0,n-1]);reference[0,2].copy_(scratch[0,0])
        torch.testing.assert_close(temporal,reference,rtol=0,atol=0)
    print('32-round native snapshot/replay outputs, states and prefix tracking: bitwise equal',flush=True)

if __name__=='__main__':test_closed_loop()
