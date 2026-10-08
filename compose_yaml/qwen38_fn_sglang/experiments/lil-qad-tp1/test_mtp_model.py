"""Construct and load the whole draft module before a full serving trial."""
import json,pathlib,sys
import torch
from safetensors import safe_open
from types import SimpleNamespace
from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
from sglang.srt.distributed import init_distributed_environment,initialize_model_parallel
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from sglang.srt.model_loader.weight_utils import get_quant_config
from qad_moe import QADW4A16MoEMethod
p=pathlib.Path(sys.argv[1])
args=ServerArgs(model_path=str(p),disable_cuda_graph=True,quantization='modelopt_mixed',
    moe_runner_backend='flashinfer_cutlass',disable_shared_experts_fusion=True)
set_global_server_args_for_scheduler(args);initialize_moe_config(args)
init_distributed_environment(1,0,'tcp://127.0.0.1:29588',0,'nccl');initialize_model_parallel(1)
torch.set_default_dtype(torch.bfloat16)
config=Qwen4ExpConfig.from_pretrained(p,local_files_only=True)
from sglang.srt.layers.dp_attention import initialize_dp_attention
initialize_dp_attention(args,SimpleNamespace(hf_config=config,hidden_size=config.text_config.hidden_size,dtype=torch.bfloat16))
quant=get_quant_config(SimpleNamespace(model_path=str(p),quantization='modelopt_mixed',
    hf_config=config,is_draft_model=True,is_draft_quantization_explicit=False),
    SimpleNamespace(download_dir=None),Qwen4ExpForConditionalGeneration.packed_modules_mapping)
with torch.device('cuda'):
    model=Qwen4ExpForCausalLMMTP(config,quant)
idx=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
keys=[k for k in idx if k.startswith('mtp.')]
def weights():
    for filename in sorted({idx[k] for k in keys}):
        with safe_open(p/filename,framework='pt',device='cpu') as f:
            for key in keys:
                if idx[key]==filename:yield key,f.get_tensor(key)
model.load_weights(weights())
count=0
for module in model.modules():
    method=getattr(module,'quant_method',None)
    if method is not None:
        method.process_weights_after_loading(module)
        if isinstance(method,QADW4A16MoEMethod):
            assert method.quant_mode=='w4a16'
            count+=1
assert count==1,count
from sglang.srt.layers.moe.topk import StandardTopKOutput
with torch.no_grad():
    experts=model.model.layers[0].mlp.experts
    x=torch.randn((4,2560),device='cuda',dtype=torch.bfloat16)*0.1
    ids=torch.tensor([[0,7,63,127,255,300,400,450,500,511]]*4,device='cuda',dtype=torch.int32)
    weights=torch.full((4,10),0.1,device='cuda',dtype=torch.float32)
    output=experts(x,StandardTopKOutput(weights,ids,torch.zeros((4,512),device='cuda')))
    assert output.shape==x.shape and torch.isfinite(output).all()
    print('Full 512-expert MTP dispatch/combine forward passed',flush=True)
print(f'Whole MTP loaded: {len(keys)} checkpoint tensors, {count} W4A16 expert layer',flush=True)
torch.distributed.destroy_process_group()
