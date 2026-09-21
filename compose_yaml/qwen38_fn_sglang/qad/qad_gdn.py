"""Opt-in TP1 B12X GDN adapters; SGLang retains convolution and commit logic.

The speculative checkpoint views alias one allocation. B12X verify reads the
persistent slot in the last index column and writes only intermediate columns.
SGLang owns accepted-token commit: scatter for the default snapshots, or
its native fold for the opt-in ReplaySSM verify path.
"""
from dataclasses import replace
import os
import torch
import triton
import triton.language as tl

ENABLED = os.environ.get('SGLANG_QAD_B12X_GDN') == '1'
_plans = {}
_workspaces = {}


@triton.jit
def _copy_checkpoint(state, source, destination, active,
                     ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    seq=tl.program_id(0)
    enabled=tl.load(active+seq)
    src=tl.load(source+seq)
    dst=tl.load(destination+seq)
    i=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(state+src*ELEMENTS+i,enabled & (src!=0) & (i<ELEMENTS),0.0)
    tl.store(state+dst*ELEMENTS+i,value,enabled & (i<ELEMENTS))


def _copy_tracked_state(state, source, destination, active):
    elements=state[0].numel()
    _copy_checkpoint[(source.numel(),triton.cdiv(elements,1024))](
        state,source,destination,active,elements,1024)


def join_state_pool(pool):
    state=pool.mamba_cache
    temporal=state.temporal
    intermediate=getattr(state,'intermediate_ssm',None)
    replay = getattr(state, 'replayssm_rawv', None) is not None
    if getattr(state, 'replayssm_d', None) is not None:
        raise ValueError('QAD B12X GDN does not support the decode ReplaySSM ring')
    if replay and (os.environ.get('SGLANG_QAD_REPLAY_VERIFY') != '1'
                   or not getattr(pool, 'replayssm_spec_fold', False)
                   or getattr(pool, 'replayssm_is_kda', False)
                   or intermediate is not None):
        raise ValueError('QAD replay verify requires opt-in GDN fold-every-commit without snapshots')
    if temporal.dtype!=torch.float32 or temporal.shape[-3:]!=(48,128,128):
        raise ValueError('QAD B12X GDN is qualified for TP1 48x128x128 FP32 state')
    pool.qad_persistent_slots=temporal.shape[1]
    if intermediate is None:
        pool.qad_state_pool=temporal
        return
    layers,rows,steps,*shape=intermediate.shape
    if steps>7:
        raise ValueError('B12X verify needs one extra column for the persistent source')
    joined=torch.empty((layers,temporal.shape[1]+rows*steps,*shape),
                       device=temporal.device,dtype=temporal.dtype)
    persistent=joined[:,:temporal.shape[1]]
    scratch=joined[:,temporal.shape[1]:].view(intermediate.shape)
    persistent.copy_(temporal);scratch.copy_(intermediate)
    pool.qad_state_pool=joined
    pool.mamba_cache=replace(state,temporal=persistent,intermediate_ssm=scratch)


def _workspace(api,kind,caps):
    key=(kind,caps)
    if key not in _plans:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('B12X GDN plan must be warmed before CUDA capture')
        _plans[key]=api.plan(caps)
    plan=_plans[key]
    work_key=(key,torch.cuda.current_stream().cuda_stream)
    if work_key not in _workspaces:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('B12X GDN scratch must be warmed on the capture stream')
        _workspaces[work_key]=[
            torch.empty(spec.shape,dtype=spec.dtype,device=caps.device)
            for spec in plan.scratch_specs()]
    return plan,_workspaces[work_key]


def recurrent(mixed_qkv,a,b,z,norm_weight,A_log,dt_bias,state,indices,cu,
              *,gate='sigmoid',eps=1e-6):
    """Decode or linear-chain verify, with an explicit checkpoint index matrix."""
    from b12x.sequence import gdn_decode as api
    tokens=mixed_qkv.shape[0];seqs=indices.shape[0];columns=indices.shape[1]
    hv=state.shape[1];hk=(mixed_qkv.shape[1]-hv*128)//256
    if (hk,hv)!=(16,48):raise ValueError('Unexpected TP1 GDN geometry')
    caps=api.Caps(device=mixed_qkv.device,max_tokens=tokens,max_seqs=seqs,
        max_state_slots=state.shape[0],key_heads=hk,value_heads=hv,
        state_index_columns=columns,state_dtype=state.dtype,gate_activation=gate,
        null_state_index=-1)
    plan,scratch=_workspace(api,'decode',caps)
    output=torch.empty((tokens,hv,128),device=mixed_qkv.device,dtype=mixed_qkv.dtype)
    binding=api.bind(plan,scratch=scratch,mixed_qkv=mixed_qkv,a=a,b=b,
        z=z.reshape(tokens,hv,128),A_log=A_log,dt_bias=dt_bias,norm_weight=norm_weight,
        recurrent_state=state,query_start_loc=cu.to(torch.int32),
        num_accepted_tokens=torch.full((seqs,),columns,device=cu.device,dtype=torch.int32),
        state_indices=indices.contiguous(),
        num_seqs=torch.full((1,),seqs,device=cu.device,dtype=torch.int32),
        num_tokens=torch.full((1,),tokens,device=cu.device,dtype=torch.int32),output=output)
    return api.run(binding,eps=eps).unsqueeze(0)


def decode(backend,layer,forward_batch,mixed_qkv,a,b,kwargs):
    metadata=backend.forward_metadata
    state=backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id).temporal
    slots=metadata.mamba_cache_indices[:mixed_qkv.shape[0]]
    return recurrent(mixed_qkv,a,b,kwargs['qad_z'],kwargs['qad_norm_weight'],
        layer.A_log,layer.dt_bias,state,slots[:,None],metadata.query_start_loc,
        gate=kwargs['qad_gate'],eps=kwargs['qad_eps'])


def verify(backend,layer,forward_batch,mixed_qkv,a,b,kwargs):
    if int(getattr(forward_batch.spec_info,'topk',1) or 1)!=1:
        raise ValueError('B12X GDN verify requires a linear MTP chain')
    pool=backend.req_to_token_pool.mamba_pool
    if getattr(pool, 'replayssm_spec_fold', False):
        if os.environ.get('SGLANG_QAD_REPLAY_VERIFY') != '1':
            raise ValueError('QAD replay verify is not enabled')
        from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
        q,k,v = torch.split(mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        tokens = mixed_qkv.shape[0]
        cache = backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        output = backend._replayssm_fold_target_verify(
            layer=layer, query=q.reshape(1,tokens,16,128),
            key=k.reshape(1,tokens,16,128), value=v.reshape(1,tokens,48,128),
            a=a,b=b,layer_cache=cache,ssm_states=cache.temporal,
            cache_indices=backend.forward_metadata.mamba_cache_indices,
            query_start_loc=backend.forward_metadata.query_start_loc,
            retrieve_parent_token=None)
        return rms_norm_gated(x=output.reshape(-1,128), weight=kwargs['qad_norm_weight'],
            bias=None,z=kwargs['qad_z'].reshape(-1,128),eps=kwargs['qad_eps'],
            norm_before_gate=True,is_rms_norm=True,activation=kwargs['qad_gate']).reshape(1,tokens,48,128)
    li=backend.req_to_token_pool.mamba2_layer_index(layer.layer_id)
    state=pool.qad_state_pool[li]
    steps=forward_batch.spec_info.draft_token_num
    seqs=mixed_qkv.shape[0]//steps
    current=backend.forward_metadata.mamba_cache_indices[:seqs]
    rows=backend.verify_intermediate_state_indices[:seqs]
    stride=pool.mamba_cache.intermediate_ssm.shape[2]
    destinations=pool.qad_persistent_slots+rows[:,None]*stride+torch.arange(steps,device=state.device)[None]
    indices=torch.cat((destinations,current[:,None]),dim=1)
    # Ignore graph-padding rows entirely, including their scratch destinations.
    indices=torch.where(current[:,None]>=0,indices,-1).to(torch.int64)
    return recurrent(mixed_qkv,a,b,kwargs['qad_z'],kwargs['qad_norm_weight'],
        layer.A_log,layer.dt_bias,state,indices,backend.forward_metadata.query_start_loc,
        gate=kwargs['qad_gate'],eps=kwargs['qad_eps'])


def prefill(backend,layer,forward_batch,mixed_qkv,a,b,state,slots):
    """Keep native SGLang convolution and export its one requested prefix checkpoint."""
    from b12x.sequence import gdn_prefill as api
    from sglang.srt.runtime_context import mamba_cache_chunk_size
    tokens=mixed_qkv.shape[0];seqs=slots.numel()
    if layer.num_k_heads!=16 or layer.num_v_heads!=48:
        raise ValueError('Unexpected TP1 GDN prefill geometry')
    q,k,v=torch.split(mixed_qkv,[layer.q_dim,layer.k_dim,layer.v_dim],dim=-1)
    q=q.view(tokens,16,128);k=k.view(tokens,16,128);v=v.view(tokens,48,128)
    cu=backend.forward_metadata.query_start_loc.to(torch.int32)
    initial=torch.where(forward_batch.extend_prefix_lens[:seqs]>0,slots,0).to(torch.int64)
    final=slots.to(torch.int64)
    checkpoint=torch.zeros_like(final);offsets=torch.zeros_like(final,dtype=torch.int32)
    aligned=None
    if backend.forward_metadata.has_mamba_track_mask:
        mask=forward_batch.mamba_track_mask[:seqs]
        checkpoint=torch.where(mask,forward_batch.mamba_track_indices[:seqs],0).to(torch.int64)
        desired=forward_batch.mamba_track_seqlens[:seqs]-forward_batch.extend_prefix_lens[:seqs]
        chunk=mamba_cache_chunk_size()
        aligned=mask & (desired%chunk==0)
        tracked=checkpoint
        # Native aligned tracking copies the final state. Exporting that as
        # an in-kernel checkpoint is invalid when the last live chunk has a
        # non-16-aligned tail: B12X deliberately returns NaNs on that metadata.
        offsets=(desired//chunk*chunk).to(torch.int32)
        offsets=torch.where(mask,offsets,0)
        # Checkpoint at offset zero is the incoming state, not a computed tile.
        zero=mask & ~aligned & (offsets==0)
        _copy_tracked_state(state,initial,tracked,zero)
        checkpoint=torch.where((offsets>0) & ~aligned,checkpoint,0)
        offsets=torch.where(aligned,0,offsets)
    capacity=1 << (tokens-1).bit_length()
    caps=api.Caps(device=mixed_qkv.device,max_tokens=capacity,max_seqs=seqs,
        max_state_slots=state.shape[0],key_heads=16,value_heads=48,
        checkpoint_export=True,null_state_index=0)
    plan,scratch=_workspace(api,'prefill',caps)
    # SGLang can pad physical rows beyond the cumulative live sequence length.
    # Such rows are capacity, not tokens. Keep them deterministically zero.
    output=torch.zeros_like(v)
    binding=api.bind(plan,scratch=scratch,q=q,k=k,v=v,a=a,b=b,A_log=layer.A_log,
        dt_bias=layer.dt_bias,recurrent_state=state,cu_seqlens=cu,
        initial_state_indices=initial,final_state_indices=final,
        checkpoint_state_indices=checkpoint,checkpoint_offsets=offsets,
        num_seqs=torch.full((1,),seqs,device=cu.device,dtype=torch.int32),
        num_tokens=cu[-1:].contiguous(),output=output)
    api.prewarm(binding)
    result=api.run(binding)
    torch._assert_async(binding.error_code.eq(0).all(), 'QAD GDN prefill metadata rejected')
    if aligned is not None:
        _copy_tracked_state(state,final,tracked,aligned)
    return result.unsqueeze(0)
