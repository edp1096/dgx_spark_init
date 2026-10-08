"""Isolated NVFP4 W4A4 adapter; same SGLang weight layout and static scales."""
import torch
from b12x.moe import fused_moe as b
_PLANS={}
def prepare(w1,s1,alpha1,a1,w2,s2,alpha2,a2,activation='silu'):
    plan=b.plan_weights(quant_modes=('nvfp4',),source_format='modelopt_nvfp4',activation=activation,params_dtype=torch.bfloat16,num_experts=w1.shape[0],hidden_size=w2.shape[1],intermediate_size=w2.shape[2]*2,w13_layout='w13')
    # SGLang exposes precombined runtime alphas; B12X wants weight globals.
    experts=b.prepare_weights(plan=plan,params_dtype=torch.bfloat16,w1_fp4=w1,w2_fp4=w2,w1_blockscale=s1,w2_blockscale=s2,w1_global_scale=alpha1*a1,w2_global_scale=alpha2*a2,a1_gscale=a1,a2_gscale=a2,immutable_input_scales=True)
    return plan,experts

def run(owner,x,ids,weights):
    wp,experts=owner
    key=(id(experts),wp,x.shape[0],ids.shape[1],x.device)
    if key not in _PLANS:
        assert not torch.cuda.is_current_stream_capturing(),'Unprepared B12X shape during graph capture'
        plan=b.plan(b.Caps(max_tokens=x.shape[0],num_topk=ids.shape[1],device=x.device,weight_plan=wp,quant_mode='nvfp4',core_token_counts=(x.shape[0],),route_num_experts=0))
        scratch=tuple(torch.empty(spec.shape,dtype=spec.dtype,device=spec.device) for spec in plan.scratch_specs())
        _PLANS[key]=(plan,scratch)
    plan,scratch=_PLANS[key]
    out=torch.empty_like(x)
    binding=plan.bind(scratch=scratch,a=x,experts=experts,topk_ids=ids.to(torch.int32),topk_weights=weights,output=out,input_scales_static=True)
    return b.run(binding=binding)

def apply(layer,dispatch,config):
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
    assert dispatch.hidden_states_scale is None
    assert layer.moe_ep_size==1 and not config.apply_router_weight_on_input
    if not hasattr(layer,'_b12x_probe_owner'):
        layer._b12x_probe_owner=prepare(layer.w13_weight,layer.w13_blockscale_swizzled,layer.g1_alphas,layer.w13_input_scale_quant,layer.w2_weight,layer.w2_blockscale_swizzled,layer.g2_alphas,layer.w2_input_scale_quant,config.activation)
    t=dispatch.topk_output
    return StandardCombineInput(hidden_states=run(layer._b12x_probe_owner,dispatch.hidden_states,t.topk_ids,t.topk_weights))
