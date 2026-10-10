"""Check actual converted weights and the real SM121 FP8 GEMM, not a mock."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
import hybrid_dispatch

hybrid_dispatch.install()
p=Path('/hf/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4-fp8hybrid')
args=ServerArgs(model_path=str(p),disable_cuda_graph=True,fp8_gemm_runner_backend='flashinfer_cutlass')
set_global_server_args_for_scheduler(args)
initialize_fp8_gemm_config(args)
init_distributed_environment(1,0,'tcp://127.0.0.1:29584',0,'nccl')
initialize_model_parallel(1)
config=ModelOptFp4Config.from_config(json.loads((p/'hf_quant_config.json').read_text()))
config.packed_modules_mapping={}
index=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
torch.manual_seed(20261009)
results=[]
for suffix in ['layers.0.linear_attn.in_proj_qkv','layers.11.self_attn.k_proj','layers.0.mlp.shared_expert.down_proj']:
    name='model.language_model.'+suffix
    with safe_open(p/index[name+'.weight'],framework='pt',device='cpu') as f:
        weight=f.get_tensor(name+'.weight').cuda()
        scales=f.get_tensor(name+'.weight_scale_inv').cuda()
    rows,cols=weight.shape
    with torch.device('cuda'):
        layer=ReplicatedLinear(cols,rows,bias=False,params_dtype=torch.bfloat16,quant_config=config,prefix='model.'+suffix)
    assert type(layer.quant_method).__name__=='Fp8LinearMethod'
    with torch.no_grad():
        layer.weight.copy_(weight);layer.weight_scale_inv.copy_(scales)
        reference_weight=(weight.float()*scales.repeat_interleave(128,0).repeat_interleave(128,1)).to(torch.bfloat16)
        layer.quant_method.process_weights_after_loading(layer)
        for tokens in [1,4,128]:
            x=torch.randn(tokens,cols,device='cuda',dtype=torch.bfloat16)
            actual,_=layer(x)
            expected=x.float()@reference_weight.float().T
            relative=((actual.float()-expected).norm()/expected.norm()).item()
            assert torch.isfinite(actual).all().item() and relative<0.06,(suffix,tokens,relative)
            results.append(dict(layer=suffix,tokens=tokens,relative_l2=relative,dtype=str(layer.weight.dtype)))
            torch.cuda.synchronize()
print('LINEAR_PROBE '+json.dumps(results),flush=True)
torch.distributed.destroy_process_group()
