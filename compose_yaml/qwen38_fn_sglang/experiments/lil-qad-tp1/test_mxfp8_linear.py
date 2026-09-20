import json
import pathlib
import sys
import torch
from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.layers.linear import ReplicatedLinear
set_global_server_args_for_scheduler(ServerArgs(model_path='/metadata',disable_cuda_graph=True))
c=ModelOptMixedPrecisionConfig.from_config(json.load(open('/metadata/hf_quant_config.json')))
from sglang.srt.distributed import init_distributed_environment,initialize_model_parallel
init_distributed_environment(1,0,'tcp://127.0.0.1:29583',0,'nccl')
initialize_model_parallel(1)
c.packed_modules_mapping={}
with torch.device('cuda'):
    layer=ReplicatedLinear(2560,96,bias=False,params_dtype=torch.bfloat16,quant_config=c,prefix='model.language_model.layers.0.linear_attn.in_proj_a')
print(type(layer.quant_method).__name__,[(n,str(p.dtype),list(p.shape)) for n,p in layer.named_parameters()],flush=True)

torch.manual_seed(18)
with torch.no_grad():
    if len(sys.argv)>1:
        from safetensors import safe_open
        snapshot=pathlib.Path(sys.argv[1])
        index=json.loads((snapshot/'model.safetensors.index.json').read_text())['weight_map']
        weights=[];scales=[]
        for part in ('b','a'):
            prefix=f'model.language_model.layers.0.linear_attn.in_proj_{part}'
            for suffix,values in (('weight',weights),('weight_scale',scales)):
                key=f'{prefix}.{suffix}'
                with safe_open(snapshot/index[key],framework='pt',device='cpu') as f:
                    values.append(f.get_tensor(key))
        layer.weight.copy_(torch.cat(weights))
        layer.weight_scale_inv.copy_(torch.cat(scales))
        exponent=layer.weight_scale_inv.int()-127
        reference_weight=layer.weight.float()*torch.exp2(exponent.float()).repeat_interleave(32,dim=1)
        print('Using actual checkpoint layer 0 fused b/a weights and E8M0 scales',flush=True)
    else:
        layer.weight.copy_((torch.randn_like(layer.weight,dtype=torch.float32)*0.02).to(torch.float8_e4m3fn))
        layer.weight_scale_inv.fill_(127)
        reference_weight=layer.weight.float().clone()
    layer.quant_method.process_weights_after_loading(layer)
    x=torch.randn((2,2560),device='cuda',dtype=torch.bfloat16)
    actual,_=layer(x)
    expected=x.float() @ reference_weight.T
    relative=((actual.float()-expected).norm()/expected.norm()).item()
    assert torch.isfinite(actual).all().item() and relative < 0.1,relative
    print('MXFP8 dense forward relative L2 error:',relative,flush=True)
torch.distributed.destroy_process_group()
