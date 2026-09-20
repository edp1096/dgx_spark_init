"""Actual text experts: pinned B12X NVFP4 vs the qualified CUTLASS path."""
import copy,json,pathlib,sys
import torch
from safetensors import safe_open
from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
from sglang.srt.distributed import init_distributed_environment,initialize_model_parallel
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig, ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from qad_moe import QADNVFP4MoEMethod
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
p=pathlib.Path(sys.argv[1])
args=ServerArgs(model_path=str(p),disable_cuda_graph=True,
    quantization='modelopt_mixed',moe_runner_backend='flashinfer_cutlass')
set_global_server_args_for_scheduler(args)
from sglang.srt.layers.moe.utils import initialize_moe_config
initialize_moe_config(args)
init_distributed_environment(1,0,'tcp://127.0.0.1:29587',0,'nccl');initialize_model_parallel(1)
torch.set_default_dtype(torch.bfloat16)
q=ModelOptMixedPrecisionConfig.from_config(json.loads((p/'hf_quant_config.json').read_text()))
q.packed_modules_mapping={}
q.get_quant_method=lambda layer,prefix:ModelOptNvFp4FusedMoEMethod(q.nvfp4_config)
b=copy.copy(q)
b.get_quant_method=lambda layer,prefix:QADNVFP4MoEMethod(q.nvfp4_config)
with torch.device('cuda'):
    layers=[FusedMoE(num_experts=2,hidden_size=2560,intermediate_size=640,layer_id=0,top_k=2,
        params_dtype=torch.bfloat16,quant_config=quant,prefix='model.language_model.layers.0.mlp.experts') for quant in (q,b)]
index=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
references=[]
for e in range(2):
    refs={}
    for proj,destination,shard in (('gate_proj','w13','w1'),('up_proj','w13','w3'),('down_proj','w2','w2')):
        data={}
        for suffix in ('weight','weight_scale','weight_scale_2','input_scale'):
            key=f'model.language_model.layers.0.mlp.experts.{e}.{proj}.{suffix}'
            with safe_open(p/index[key],framework='pt',device='cpu') as f:v=f.get_tensor(key)
            data[suffix]=v
            for layer in layers:
                name=f'{destination}_{suffix}'
                param=getattr(layer,name)
                param.weight_loader(param,v,name,shard_id=shard,expert_id=e)
        w=data['weight']
        refs[proj]=NVFP4QTensor(torch.Size((w.shape[0],w.shape[1]*2)),torch.bfloat16,w).dequantize(
            scale=data['weight_scale'],double_scale=data['weight_scale_2'],block_sizes={-1:16}).cuda()
    references.append(refs)
for layer in layers:layer.quant_method.process_weights_after_loading(layer)
with torch.no_grad():
    torch.manual_seed(19)
    for m in (1,4,17,64):
        x=torch.randn((m,2560),device='cuda',dtype=torch.bfloat16)*0.1
        ids=torch.arange(2,device='cuda',dtype=torch.int32).expand(m,2).contiguous()
        weights=torch.softmax(torch.randn((m,2),device='cuda',dtype=torch.float32),dim=-1)
        dispatch=StandardDispatchOutput(x,None,StandardTopKOutput(weights,ids,None))
        reference=layers[0].quant_method.apply(layers[0],dispatch).hidden_states
        actual=layers[1].quant_method.apply(layers[1],dispatch).hidden_states
        relative=((actual.float()-reference.float()).norm()/reference.float().norm()).item()
        bf16=torch.zeros_like(x,dtype=torch.float32)
        for e,r in enumerate(references):
            hidden=torch.nn.functional.silu(x @ r['gate_proj'].T)*(x @ r['up_proj'].T)
            bf16+=(hidden @ r['down_proj'].T).float()*weights[:,e,None]
        cutlass_error=((reference.float()-bf16).norm()/bf16.norm()).item()
        b12x_error=((actual.float()-bf16).norm()/bf16.norm()).item()
        print(f'Actual text NVFP4 experts M={m}: B12X/CUTLASS relative L2={relative}; against BF16 activations CUTLASS={cutlass_error}, B12X={b12x_error}',flush=True)
        assert torch.isfinite(actual).all(),relative
        # Require comparable activation-quantization error, not artificial
        # bitwise equality between different fused low-precision kernels.
        assert relative<0.06 and b12x_error<cutlass_error+0.01 and b12x_error<0.20
    from graph_probe import check_replay
    check_replay(layers[1], dispatch)
torch.distributed.destroy_process_group()
