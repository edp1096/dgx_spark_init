import json,math,time,sys
import torch
from glm53_b12x import SparseAttention

def pack(x):
    x=x.float().reshape(-1,4,128)
    scales=x.abs().amax(-1).clamp_min(1e-12)/448
    quant=(x/scales[...,None]).to(torch.float8_e4m3fn)
    raw=torch.cat([quant.reshape(-1,512).view(torch.uint8),scales.contiguous().view(torch.uint8).reshape(-1,16)],-1)
    restored=(quant.float()*scales[...,None]).reshape(-1,512)
    return raw[:,None,:].contiguous(),restored

torch.manual_seed(41)
results=[]
for rows,heads,tail in [(1,32,0),(1,32,3),(7,32,3),(1,64,3)]:
    torch.cuda.synchronize();start=time.monotonic()
    values=torch.randn(4096,512,device='cuda',dtype=torch.bfloat16)*.25
    values[-3:,0]=8
    cache,dequant=pack(values)
    if '--sglang-writer' in sys.argv:
        from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate
        native,rope=quantize_k_cache_separate(values[:,None,:],None)
        assert rope.numel()==0 and native.shape[-1]==528
        quant=native[:,0,:512].contiguous().view(torch.float8_e4m3fn).float().reshape(-1,4,128)
        scales=native[:,0,512:].contiguous().view(torch.float32).reshape(-1,4,1)
        dequant=(quant*scales).reshape(-1,512)
        cache=native.contiguous()
    query=torch.randn(rows,heads,512,device='cuda',dtype=torch.bfloat16)*.25
    query[:,:,0]=8
    indices=torch.full((rows,2051),-1,device='cuda',dtype=torch.int32)
    for i in range(rows):
        n=71+i*7
        indices[i,:n]=torch.randperm(3000,device='cuda')[:n].int()
        if tail:indices[i,2048:]=torch.arange(4093,4096,device='cuda',dtype=torch.int32)
    attn=SparseAttention();out=attn(query,cache,indices,scale=1/16).clone()
    reference=[]
    for i in range(rows):
        selected=dequant[indices[i][indices[i]>=0].long()]
        reference.append(torch.softmax(query[i].float()@selected.T/16,dim=-1)@selected)
    ref=torch.stack(reference)
    torch.testing.assert_close(out.float(),ref,atol=.03,rtol=.03)
    repeated=attn(query,cache,indices,scale=1/16)
    torch.testing.assert_close(out,repeated,atol=0,rtol=0)
    torch.cuda.synchronize()
    result={'rows':rows,'heads':heads,'tail':tail,'max_error':(out.float()-ref).abs().max().item(),'seconds':time.monotonic()-start}
    print(json.dumps(result),flush=True);results.append(result)
print('PASS',len(results),flush=True)

if '--cuda-graph' in sys.argv:
    # Both ordinary decode and DFlash K5 verify (6 rows) must survive prefill.
    base_query=query[:1].contiguous();base_indices=indices[:1].contiguous()
    for graph_rows in (1,6):
        query=base_query.repeat(graph_rows,1,1);indices=base_indices.repeat(graph_rows,1)
        attn=SparseAttention()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): eager=attn(query,cache,indices,scale=1/16)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream): captured=attn(query,cache,indices,scale=1/16)
        torch.cuda.current_stream().wait_stream(stream)
        for step in range(3):
            attn(base_query.repeat(7,1,1),cache,base_indices.repeat(7,1),scale=1/16)
            query.mul_(.8);indices[:,0]=100+step;eager=attn(query,cache,indices,scale=1/16).clone()
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(captured,eager,atol=0,rtol=0)
        print('PASS graph rows',graph_rows,flush=True)
    print('PASS sparse+graph replay after prefill',len(results),flush=True)
