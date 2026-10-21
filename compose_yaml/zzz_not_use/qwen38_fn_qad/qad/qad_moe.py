"""TP1 SGLang W4A16 dispatch into the pinned B12X expert kernel API."""
import torch
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod, deinterleave_w13
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

# Same-stream launches execute sequentially; different CUDA streams never share scratch.
_scratch = {}


class QADW4A16MoEMethod(ModelOptNvFp4FusedMoEMethod):
    quant_mode = 'w4a16'
    @property
    def load_up_proj_weight_first(self):
        return False  # B12X is explicitly told [gate, up].

    def create_moe_runner(self, layer, moe_runner_config):
        if layer.moe_tp_size != 1 or layer.moe_ep_size != 1:
            raise ValueError('QAD B12X expert adapter currently requires TP1/EP1')
        if moe_runner_config.activation != 'silu' or not moe_runner_config.is_gated:
            raise ValueError('QAD expert adapter requires gated SiLU')
        if moe_runner_config.apply_router_weight_on_input:
            raise ValueError('QAD expert adapter requires output-side router weighting')
        self.moe_runner_config=moe_runner_config

    def process_weights_after_loading(self, layer):
        from b12x.moe import fused_moe as moe
        from b12x._lib.intrinsics import swizzle_block_scale
        if getattr(layer,'inference_moe_w13_interleaved',False):
            layer.w13_weight.data=deinterleave_w13(layer.w13_weight.data)
            layer.w13_weight_scale.data=deinterleave_w13(layer.w13_weight_scale.data)
        gs=layer.w13_weight_scale_2.detach()
        if gs.ndim!=2 or gs.shape[1]!=2 or not torch.equal(gs[:,0],gs[:,1]):
            raise ValueError('B12X W4A16 adapter requires equal gate/up global scales')
        if not torch.isfinite(gs).all() or not (gs>0).all():
            raise ValueError('Invalid W4A16 global scales')
        e,n,k=layer.w13_weight.shape
        extra={'w4a16_layout':'mma_packed'} if self.quant_mode=='w4a16' else {}
        self._plan=moe.plan_weights(quant_modes=self.quant_mode,source_format='modelopt_nvfp4',
            activation='silu',params_dtype=torch.bfloat16,num_experts=e,
            # B12X calls physical [gate, up] "w31" (SGLang calls it w13).
            hidden_size=k*2,intermediate_size=n//2,w13_layout='w31',**extra)
        ones=torch.ones(e,device=gs.device,dtype=torch.float32)
        a1,a2=ones,ones
        if self.quant_mode=='nvfp4':
            a1=layer.w13_input_scale.detach().float().max().reciprocal().expand(e).contiguous()
            a2=layer.w2_input_scale.detach().float().max().reciprocal().expand(e).contiguous()
            if not torch.isfinite(a1).all() or not torch.isfinite(a2).all() or not (a1>0).all() or not (a2>0).all():
                raise ValueError('Invalid NVFP4 activation calibration scales')
        self._experts=moe.prepare_weights(plan=self._plan,params_dtype=torch.bfloat16,
            w1_fp4=layer.w13_weight.detach(), w2_fp4=layer.w2_weight.detach(),
            w1_global_scale=gs[:,0].contiguous(),w2_global_scale=layer.w2_weight_scale_2.detach(),
            w1_blockscale=swizzle_block_scale(layer.w13_weight_scale.detach()),
            w2_blockscale=swizzle_block_scale(layer.w2_weight_scale.detach()),
            a1_gscale=a1,a2_gscale=a2,immutable_input_scales=True)
        # Keep module parameters as aliases of B12X's canonical allocations.
        # Otherwise every layer retains raw block scales beside swizzled ones
        # (7.03 GiB across 48 target layers), and MTP retains its source weights
        # beside the MMA repack. Preserve loader metadata and logical shapes.
        for name, field in (('w13_weight','w1_fp4'), ('w2_weight','w2_fp4'),
                            ('w13_weight_scale','w1_blockscale'),
                            ('w2_weight_scale','w2_blockscale')):
            original=getattr(layer,name)
            canonical=getattr(self._experts,field)
            if not canonical.is_contiguous() or canonical.nbytes!=original.nbytes:
                raise ValueError(f'Unexpected B12X canonical storage geometry: {name}')
            view=canonical.view(original.dtype).view(original.shape)
            original.data=view
        for name in ('w13_blockscale_swizzled','w2_blockscale_swizzled'):
            if hasattr(layer,name):
                delattr(layer,name)
        self._plans={}
        print(f'QAD_B12X_MOE_READY mode={self.quant_mode} experts={e} hidden={k*2} intermediate={n//2}',flush=True)

    def apply(self, layer, dispatch_output):
        from b12x.moe import fused_moe as moe
        x=dispatch_output.hidden_states
        if x.dtype!=torch.bfloat16 or dispatch_output.hidden_states_scale is not None:
            raise ValueError('W4A16 requires unquantized BF16 activations')
        ids=dispatch_output.topk_output.topk_ids.contiguous()
        weights=dispatch_output.topk_output.topk_weights.contiguous()
        if x.shape[0]==0:
            return StandardCombineInput(torch.empty_like(x))
        capacity=1 << (x.shape[0]-1).bit_length()
        key=(capacity,ids.shape[1])
        if key not in self._plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('B12X W4A16 capacity must be warmed before graph capture')
            caps=moe.Caps(max_tokens=capacity,num_topk=ids.shape[1],device=x.device,
                          weight_plan=self._plan,quant_mode=self.quant_mode,deterministic_output=True)
            self._plans[key]=(moe.plan(caps),moe.required_nbytes(caps))
        plan,nbytes=self._plans[key]
        scratch_key=(str(x.device),torch.cuda.current_stream().cuda_stream,nbytes)
        if scratch_key not in _scratch:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('B12X W4A16 scratch must be warmed before graph capture')
            _scratch[scratch_key]=torch.empty(nbytes,device=x.device,dtype=torch.uint8)
        # Let PyTorch own the output allocation (including its graph pool).
        # B12X does not allocate an implicit output during CUDA capture.
        output=torch.empty_like(x, memory_format=torch.contiguous_format)
        binding=moe.bind(plan,scratch=_scratch[scratch_key],a=x.contiguous(),experts=self._experts,output=output,
                         topk_weights=weights,topk_ids=ids,unit_scale_contract=self.quant_mode=='w4a16',
                         input_scales_static=self.quant_mode=='nvfp4')
        return StandardCombineInput(moe.run(binding=binding))


class QADNVFP4MoEMethod(QADW4A16MoEMethod):
    quant_mode = 'nvfp4'
