"""Isolated SM121 correctness and timing, no weights or serving traffic.

Import shims cover only SGLang utility helpers, not attention mathematics.
Baseline extend source is copied unchanged from the pinned serving image.
"""
import sys,types,importlib.util,json,time,math
from pathlib import Path
import torch,triton
r=Path(__file__).parent
def stub(name,**items):
    m=types.ModuleType(name);m.__dict__.update(items);sys.modules[name]=m
stub('sglang.srt.utils',is_hip=lambda:False,is_cuda=lambda:True,is_gfx95_supported=lambda:False)
stub('sglang.srt.environ',envs=None)
def strides(x,page_size):
    assert page_size==1 and x.ndim==3
    return x.stride(0),x.stride(1),0,0
stub('sglang.kernels.ops.attention.decode_attention',_extract_kv_strides=strides)
def unused(*a,**kw):raise AssertionError('unexpected shim invocation')
stub('sglang.kernels.ops.attention.prefill_attention',context_attention_fwd=unused)
def unpack(score,aux):
    assert score is None and aux is None
    return None,0,0,0
stub('sglang.kernels.ops.attention.score_mod',unpack_aux_tensors=unpack)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
new=load('verify_sm121',r/'verify_splitkv.py')
base=load('extend_pinned',r/'extend_attention.py')
assert torch.cuda.get_device_capability()==(12,1)
torch.manual_seed(4731)
def make(hkv,d,width,lengths,window):
    bs=len(lengths);pool=sum(lengths)+41
    q=torch.randn(bs*width,32,d,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(bs*width,hkv,d,device='cuda',dtype=torch.bfloat16)
    v=k if d==512 else torch.randn_like(k)
    kb=torch.randn(pool,hkv,d,device='cuda',dtype=torch.bfloat16)
    vb=kb if d==512 else torch.randn_like(kb)
    qi=torch.arange(bs+1,device='cuda',dtype=torch.int32)*width
    ki=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int32)
    ix=torch.randperm(pool,device='cuda')[:sum(lengths)]
    out=torch.empty_like(q)
    return q,k,v,kb,vb,qi,ki,ix,out
def reference(t,lengths,width,window):
    q,k,v,kb,vb,qi,ki,ix,out=t;pieces=[];start=0
    for b,n in enumerate(lengths):
        kk=torch.cat([kb[ix[start:start+n]],k[b*width:(b+1)*width]]).float().repeat_interleave(32//k.shape[1],dim=1)
        vv=torch.cat([vb[ix[start:start+n]],v[b*width:(b+1)*width]]).float().repeat_interleave(32//k.shape[1],dim=1)
        qq=q[b*width:(b+1)*width].float()
        logits=torch.einsum('qhd,khd->hqk',qq,kk)/math.sqrt(q.shape[-1])
        pos=torch.arange(n+width,device='cuda')[None,:]
        end=n+torch.arange(width,device='cuda')[:,None]
        mask=pos<=end
        if window>0:mask &= pos+window>=end
        logits.masked_fill_(~mask[None,:,:],float('-inf'))
        pieces.append(torch.einsum('hqk,khd->qhd',logits.softmax(-1),vv))
        start+=n
    return torch.cat(pieces)
def baseline(t,width,window):
    q,k,v,kb,vb,qi,ki,ix,out=t
    base.extend_attention_fwd(q,k,v,out,kb,vb,qi,ki,ix,None,True,None,width,1.,1.,sliding_window_size=window)
    return out
def diff(actual,ref):
    a=actual.float();delta=(a-ref).abs()
    return {'max_abs':delta.max().item(),'rms':delta.square().mean().sqrt().item(),
            'relative_l2':((a-ref).norm()/ref.norm().clamp_min(1e-10)).item()}
def bench(fn):
    for _ in range(3):fn()
    return float(triton.testing.do_bench_cudagraph(fn,rep=100))
rows=[]
for hkv,d,window in [(16,256,1024),(4,512,-1)]:
 for width in [4,6,8]:
  for lengths in ([0],[17],[1025],[8193],[37,1027]):
    t=make(hkv,d,width,lengths,window);ref=reference(t,lengths,width,window)
    b=baseline(t,width,window).clone();bd=diff(b,ref)
    assert bd['max_abs']<.04 and bd['relative_l2']<.015,('baseline',bd)
    candidates=[]
    for ns,bn,nw in [(4,32,4),(8,32,4),(16,32,4),(8,16,4),(8,64,4),(8,32,8)]:
      vk=new.VerifySplitKV(len(lengths),32,hkv,d,d,width,n_splits=ns,block_n=bn,num_warps=nw)
      def call():return vk(*t[:8],1/math.sqrt(d),o_out=t[8],sliding_window_size=window)
      result=call();delta=diff(result,ref)
      assert delta['max_abs']<.04 and delta['relative_l2']<.015,('new',d,width,lengths,ns,bn,nw,delta)
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):call()
      t[0].mul_(.91);graph.replay();torch.cuda.synchronize()
      changed=diff(t[8],reference(t,lengths,width,window))
      assert changed['max_abs']<.04 and changed['relative_l2']<.015,('graph',changed)
      t[0].div_(.91);ref=reference(t,lengths,width,window)
      ms=bench(call) if len(lengths)==1 and lengths[0] in [17,1025,8193] else None
      candidates.append(dict(splits=ns,block_n=bn,warps=nw,ms=ms,**delta,graph=changed))
    bm=bench(lambda:baseline(t,width,window)) if len(lengths)==1 and lengths[0] in [17,1025,8193] else None
    row=dict(hq=32,hkv=hkv,dim=d,width=width,lengths=lengths,window=window,baseline=bd,baseline_ms=bm,candidates=candidates)
    rows.append(row);(r/'sm121-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(row),flush=True)
(r/'SM121_COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z')+'\n')
