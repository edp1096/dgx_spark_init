"""Experimental B12X MoE backend for TensorFold native, same-device buffers."""
import os, json, pathlib, ctypes, gc, time, sys
sys.path.insert(0,"/bridge/vendor")
os.environ["PYTHONPATH"]="/bridge/vendor:/opt/tensorfold/python"
import numpy as np
import torch
import triton
import triton.language as tl
from safetensors import safe_open
import b12x
from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe._compat import Caps
ROOT=pathlib.Path(os.environ.get("TF_B12X_MODEL", "/model"))
INDEX=json.loads((ROOT/"model.safetensors.index.json").read_text())["weight_map"]
LAYERS={}; RETAIN=[]; PLANS={}
torch.set_grad_enabled(False)
HEADERS={};FDS={}
def raw(name):
    f=ROOT/INDEX[name]
    if f not in HEADERS:
        import struct
        fd=os.open(f,os.O_RDONLY);n=struct.unpack("<Q",os.pread(fd,8,0))[0]
        HEADERS[f]=(n+8,json.loads(os.pread(fd,n,8)));FDS[f]=fd
    offset,header=HEADERS[f];entry=header[name];a,z=entry["data_offsets"]
    data=bytearray(os.pread(FDS[f],z-a,offset+a))
    assert len(data)==z-a
    types={"U8":torch.uint8,"U32":torch.uint32,"F32":torch.float32,"BF16":torch.bfloat16,"F16":torch.float16,"F8_E4M3":torch.float8_e4m3fn}
    dtype=types[entry["dtype"]]
    value=torch.frombuffer(data,dtype=dtype).reshape(entry["shape"]) if data else torch.empty(entry["shape"],dtype=dtype)
    os.posix_fadvise(FDS[f],offset+a,z-a,os.POSIX_FADV_DONTNEED)
    return value
def scales(base,n,k):
    key=base+".weight_scale"
    if key in INDEX:return raw(key).view(torch.uint8)
    f=raw(key+".nvfp4_csf_fixed").numpy().reshape(n//16,-1)
    assert f.shape[1]==16*(1+k//32) and (f[:,:16]<=240).all()
    nib=f[:,16:].reshape(n,k//32);basev=f[:,:16].reshape(n,1)
    out=np.empty((n,k//16),dtype=np.uint8)
    out[:,0::2]=basev+(nib&15);out[:,1::2]=basev+(nib>>4)
    ex=raw(key+".nvfp4_csf_exceptions").numpy().astype(np.uint32)
    pos=ex&0xffffff
    assert (pos<out.size).all()
    out.reshape(-1)[pos]=ex>>24
    return torch.from_numpy(out)
def load(name):
    t=time.monotonic();cfg=json.loads((ROOT/"config.json").read_text());cfg=cfg.get("text_config",cfg)
    E=cfg["num_experts"];H=cfg["hidden_size"];N=cfg["moe_intermediate_size"]
    # Use one CPU projection stack at a time; no second resident copy in TensorFold.
    values=[];block=[];global_s=[]
    for proj,n,k in [("gate_proj",N,H),("up_proj",N,H),("down_proj",H,N)]:
        ws=torch.empty((E,n,k//2),dtype=torch.uint8)
        ss=torch.empty((E,n,k//16),dtype=torch.uint8)
        gs=torch.empty(E,dtype=torch.float32)
        for e in range(E):
            key=f"{name}.experts.{e}.{proj}"
            ws[e].copy_(raw(key+".weight"));ss[e].copy_(scales(key,n,k));gs[e]=raw(key+".weight_scale_2")
        values.append(ws.cuda());block.append(ss.cuda().view(torch.float8_e4m3fn));global_s.append(gs.cuda())
        del ws,ss,gs
    def swizzle(t):
        e,n,k=t.shape
        assert n%128==0 and k%4==0
        return t.view(torch.uint8).reshape(e,n//128,4,32,k//4,4).permute(0,1,4,3,2,5).contiguous().reshape(e,n,k).view(torch.float8_e4m3fn)
    w13_scales=swizzle(torch.cat(block[:2],dim=1));w2_scales=swizzle(block[2])
    assert torch.equal(global_s[0],global_s[1]), "Gate/up global scales must match losslessly"
    layout=os.environ.get("TF_B12X_LAYOUT","source_native")
    plan=moe.plan_weights(quant_modes=("nvfp4","w4a16") if layout=="source_native" else "w4a16",source_format="modelopt_nvfp4",activation="silu",params_dtype=torch.bfloat16,num_experts=E,hidden_size=H,intermediate_size=N,w13_layout="w31",w4a16_layout=layout)
    ex=moe.prepare_weights(plan=plan,params_dtype=torch.bfloat16,w1_fp4=torch.cat(values[:2],dim=1),w2_fp4=values[2],w1_blockscale=w13_scales,w2_blockscale=w2_scales,w1_global_scale=global_s[0],w2_global_scale=global_s[2],a1_gscale=torch.ones(E,device="cuda"),a2_gscale=torch.ones(E,device="cuda"))
    handle=len(LAYERS)+1;LAYERS[handle]=(ex,plan,H,N,E)
    del values,block,global_s;gc.collect();torch.cuda.empty_cache()
    print(f"B12X loaded {name} handle={handle} in {time.monotonic()-t:.2f}s",flush=True)
    return handle
# __cuda_array_interface__ provides non-owning views of TensorFold allocations.
class DeviceView:
    def __init__(self,ptr,shape,typestr,strides=None):
        self.__cuda_array_interface__={"shape":tuple(shape),"typestr":typestr,"data":(ptr,False),"version":3,"strides":strides}
def view(ptr,shape,dtype,strides=None):
    types={torch.uint16:"<u2",torch.int32:"<i4",torch.float32:"<f4"}
    return torch.as_tensor(DeviceView(ptr,shape,types[dtype],strides),device="cuda")
@triton.jit
def prepare_routes(P,W,PI,WI,R:tl.constexpr,T:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    pos=(i//T)*S+i%T
    tl.store(PI+i,tl.load(P+pos,i<R*T,0),i<R*T)
    tl.store(WI+i,tl.load(W+pos,i<R*T,0),i<R*T)
@triton.jit
def normalize_output(O,Y,W,R:tl.constexpr,H:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B)
    v=tl.load(O+r*H+j,j<H,0)
    tl.store(Y+r*S*H+j,v,j<(S-1)*H)
    # Other routed slots are never consumed after their weights become zero.
    s=tl.arange(0,16)
    weights=tl.load(W+r*S+s,s<S,0)
    replacement=tl.where(s==0,1.,tl.where(s==S-1,weights,0.))
    if tl.program_id(1)==0:tl.store(W+r*S+s,replacement,s<S)
def capture(handle,stream,x,pick,wts,y,R,H,top,es):
    ex,wp,h,n,e=LAYERS[handle];assert h==H
    ext=torch.cuda.ExternalStream(stream)
    torch.cuda.synchronize()
    X=view(x,(R,H),torch.uint16).view(torch.bfloat16)
    ids=view(pick,(R,top),torch.int32,((top+1)*4,4))
    weights=view(wts,(R,top),torch.float32,((top+1)*4,4))
    assert X.data_ptr()==x and ids.data_ptr()==pick and weights.data_ptr()==wts
    W=view(wts,(R,top+1),torch.float32)
    Y=view(y,(R,(top+1)*H),torch.float32 if es==4 else torch.uint16)
    if es==2:Y=Y.view(torch.bfloat16)
    key=(R,top)
    if key not in PLANS:
        caps=Caps(max_tokens=R,num_topk=top,device=torch.device("cuda:0"),weight_plan=wp,quant_mode="w4a16",deterministic_output=True)
        p=moe.plan(caps)
        scratch={spec.name:torch.empty(spec.shape,dtype=spec.dtype,device="cuda") for spec in p.scratch_specs()}
        PLANS[key]=(p,scratch)
    p,scratch=PLANS[key]
    out=torch.empty((R,H),dtype=torch.bfloat16,device="cuda")
    ids_contiguous=torch.empty((R,top),device="cuda",dtype=torch.int32)
    weights_contiguous=torch.empty((R,top),device="cuda",dtype=torch.float32)
    binding=moe.bind(p,scratch=scratch,a=X,experts=ex,topk_weights=weights_contiguous,topk_ids=ids_contiguous,output=out)
    def run():
        prepare_routes[(triton.cdiv(R*top,256),)](ids,weights,ids_contiguous,weights_contiguous,R,top,top+1,256)
        moe.run(binding=binding)
        normalize_output[(R,triton.cdiv(H*top,256))](out,Y,W,R,H,top+1,256)
    with torch.cuda.stream(ext):
        saved_weights=W.clone()
        run();ext.synchronize()
        graph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=ext):run()
        W.copy_(saved_weights);ext.synchronize()
    RETAIN.append((graph,binding,X,ids,weights,Y,W,out,ids_contiguous,weights_contiguous))
    return graph.raw_cuda_graph()
