"""Actual TP1 geometry: scaled FP8 -> compact gather -> SM121 attention/graphs."""
import pytest
import torch
from types import SimpleNamespace

from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    _resolve_flash_attn_varlen_func, QwenSparseAttnBackend,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


@pytest.mark.parametrize('rows', [1,4])
def test_scaled_fp8_sm121_decode_and_changed_graph(rows):
    assert torch.cuda.get_device_capability()==(12,1)
    torch.manual_seed(1909+rows)
    device='cuda';topk=2052;slots=4096
    q=torch.randn(rows,24,256,device=device,dtype=torch.bfloat16)
    k=torch.randn(slots,2,256,device=device,dtype=torch.bfloat16)
    v=torch.randn_like(k)
    ks,vs=0.25,1.7
    k8=(k/ks).to(torch.float8_e4m3fn);v8=(v/vs).to(torch.float8_e4m3fn)
    mapping=torch.randperm(slots,device=device,dtype=torch.int32)[None].repeat(rows,1)
    req=torch.arange(rows,device=device,dtype=torch.int32)
    lengths=torch.full((rows,),slots,device=device,dtype=torch.int32)
    indices=torch.arange(topk,device=device,dtype=torch.int32)[None].repeat(rows,1)
    if rows>1:indices[1,-19:]=-1
    counts=torch.empty(rows,device=device,dtype=torch.int32)
    cu_k=torch.empty(rows+1,device=device,dtype=torch.int32)
    cu_q=torch.arange(rows+1,device=device,dtype=torch.int32)
    packed_k=torch.empty(rows*topk,2,256,device=device,dtype=torch.bfloat16)
    packed_v=torch.empty_like(packed_k)
    attention=_resolve_flash_attn_varlen_func()
    def execute():
        qwen_sparse_fa2_cu_seqlens_triton(lengths,indices,counts,cu_k,rows,topk)
        qwen_sparse_kv_extraction_compact_triton(k8,v8,mapping,req,indices,lengths,cu_k,
            packed_k,packed_v,rows,topk,k_scale=ks,v_scale=vs)
        return attention(q=q,k=packed_k,v=packed_v,cu_seqlens_q=cu_q,cu_seqlens_k=cu_k,
                         max_seqlen_q=1,max_seqlen_k=topk,softmax_scale=256**-.5,causal=True)
    def reference():
        out=[]
        # Round dequantized selected rows to BF16, matching the attention input contract.
        kd=(k8.float()*ks).bfloat16();vd=(v8.float()*vs).bfloat16()
        for row in range(rows):
            selected=mapping[row,indices[row][indices[row]>=0].long()].long()
            kk=kd[selected].repeat_interleave(12,dim=1).float()
            vv=vd[selected].repeat_interleave(12,dim=1).float()
            scores=torch.einsum('hd,nhd->hn',q[row].float(),kk)*(256**-.5)
            out.append(torch.einsum('hn,nhd->hd',scores.softmax(-1),vv))
        return torch.stack(out).bfloat16()
    for _ in range(3):actual=execute()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual,reference(),rtol=.03,atol=.008)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=execute()
    q.copy_(torch.randn_like(q));k8.copy_((torch.randn_like(k)/ks).to(k8.dtype))
    v8.copy_((torch.randn_like(v)/vs).to(v8.dtype));indices[:,0]=slots-1
    graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(captured,reference(),rtol=.03,atol=.008)
    torch.testing.assert_close(captured,execute(),rtol=0,atol=0)


def test_real_fp8_pool_write_preserves_inputs_and_scaled_bytes():
    pool=MHATokenToKVPool(size=128,page_size=64,dtype=torch.float8_e4m3fn,
        head_num=2,head_dim=256,layer_num=1,device='cuda',enable_memory_saver=False)
    backend=QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool=pool
    layer=SimpleNamespace(layer_id=0,k_scale_float=.25,v_scale_float=.5)
    k=torch.randn(3,2,256,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    original_k=k.clone();original_v=v.clone()
    loc=torch.tensor([7,63,127],device='cuda',dtype=torch.int64)
    backend._store_kv(layer,loc,k,v)
    torch.testing.assert_close(k,original_k,rtol=0,atol=0)
    torch.testing.assert_close(v,original_v,rtol=0,atol=0)
    torch.testing.assert_close(pool.get_key_buffer(0)[loc].float(),(k/.25).to(pool.dtype).float(),rtol=0,atol=0)
    torch.testing.assert_close(pool.get_value_buffer(0)[loc].float(),(v/.5).to(pool.dtype).float(),rtol=0,atol=0)
