"""Load actual vision weights alone and check W4A16 arithmetic plus full tower."""
import json
import pathlib
import sys
import torch
from safetensors import safe_open
from types import SimpleNamespace
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from sglang.srt.model_loader.weight_utils import get_quant_config
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

snapshot=pathlib.Path(sys.argv[1])
args=ServerArgs(model_path=str(snapshot), disable_cuda_graph=True,
                quantization='modelopt_mixed', moe_runner_backend='flashinfer_cutlass')
set_global_server_args_for_scheduler(args)
init_distributed_environment(1,0,'tcp://127.0.0.1:29585',0,'nccl')
initialize_model_parallel(1)
torch.set_default_dtype(torch.bfloat16)
config=Qwen4ExpConfig.from_pretrained(snapshot,local_files_only=True)
config.encoder_only=True
config.language_model_only=False
quant=get_quant_config(SimpleNamespace(model_path=str(snapshot),quantization='modelopt_mixed',
    hf_config=config,is_draft_model=False,is_draft_quantization_explicit=False),
    SimpleNamespace(download_dir=None),Qwen4ExpForConditionalGeneration.packed_modules_mapping)
with torch.device('cuda'):
    model=Qwen4ExpForConditionalGeneration(config,quant)
index=json.loads((snapshot/'model.safetensors.index.json').read_text())['weight_map']
keys=[k for k in index if k.startswith('model.visual.')]
def weights():
    for filename in sorted({index[k] for k in keys}):
        with safe_open(snapshot/filename,framework='pt',device='cpu') as f:
            for key in keys:
                if index[key]==filename:
                    yield key, f.get_tensor(key)
model.load_weights(weights())
layer=model.visual.blocks[0].mlp.linear_fc2
reference=NVFP4QTensor(torch.Size((1152,4304)),torch.bfloat16,layer.weight.detach()).dequantize(
    scale=layer.weight_scale.detach(),double_scale=layer.weight_scale_2.detach(),block_sizes={-1:16})
for module in model.modules():
    method=getattr(module,'quant_method',None)
    if method is not None:
        method.process_weights_after_loading(module)
with torch.no_grad():
    torch.manual_seed(19)
    for m in (1,17,64):
        x=torch.randn((m,4304),device='cuda',dtype=torch.bfloat16)
        y,_=layer(x)
        expected=(x.float() @ reference.float().T).bfloat16()+layer.bias
        relative=((y.float()-expected.float()).norm()/expected.float().norm()).item()
        assert torch.isfinite(y).all() and relative<0.015,relative
        print(f'Actual vision W4A16 FC2: M={m}, relative L2={relative}',flush=True)
    pixels=torch.randn((16,1536),device='cuda',dtype=torch.bfloat16)
    grid=torch.tensor([[1,4,4]],dtype=torch.int64)
    output=model.visual(pixels,grid)
    assert output.shape==(4,2560) and torch.isfinite(output).all(),output.shape
    print(f'Full quantized vision tower passed: {len(keys)} tensors, output={tuple(output.shape)}',flush=True)
torch.distributed.destroy_process_group()
