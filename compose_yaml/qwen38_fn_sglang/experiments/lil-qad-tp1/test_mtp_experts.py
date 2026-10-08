"""Actual MTP experts: B12X W4A16 vs dequantized BF16 reference, no A4 conversion."""
import json,pathlib,sys
import torch
from safetensors import safe_open
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
from sglang.srt.distributed import init_distributed_environment,initialize_model_parallel
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput

p=pathlib.Path(sys.argv[1])
args=ServerArgs(model_path=str(p),disable_cuda_graph=True,
    quantization='modelopt_mixed',moe_runner_backend='flashinfer_cutlass')
set_global_server_args_for_scheduler(args)
from sglang.srt.layers.moe.utils import initialize_moe_config
initialize_moe_config(args)
init_distributed_environment(1,0,'tcp://127.0.0.1:29586',0,'nccl')
initialize_model_parallel(1)
torch.set_default_dtype(torch.bfloat16)
q=ModelOptMixedPrecisionConfig.from_config(json.loads((p/'hf_quant_config.json').read_text()))
q.packed_modules_mapping={}
with torch.device('cuda'):
    layer=FusedMoE(num_experts=2,hidden_size=2560,intermediate_size=640,layer_id=0,top_k=2,
                   params_dtype=torch.bfloat16,quant_config=q,prefix='mtp.layers.0.mlp.experts')
index=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
params=dict(layer.named_parameters())
reference=[]
for e in range(2):
    refs={}
    for proj,destination,shard in (('gate_proj','w13','w1'),('up_proj','w13','w3'),('down_proj','w2','w2')):
        data={}
        for suffix in ('weight','weight_scale','weight_scale_2'):
            key=f'mtp.layers.0.mlp.experts.{e}.{proj}.{suffix}'
            with safe_open(p/index[key],framework='pt',device='cpu') as f:
                data[suffix]=f.get_tensor(key)
            name=f'{destination}_{suffix}'
            param=params[name]
            param.weight_loader(param,data[suffix],name,shard_id=shard,expert_id=e)
        w=data['weight'];shape=(w.shape[0],w.shape[1]*2)
        refs[proj]=NVFP4QTensor(torch.Size(shape),torch.bfloat16,w).dequantize(
            scale=data['weight_scale'],double_scale=data['weight_scale_2'],block_sizes={-1:16}).cuda()
    reference.append(refs)
layer.quant_method.process_weights_after_loading(layer)
with torch.no_grad():
    torch.manual_seed(19)
    for m in (1,4,17,64):
        x=torch.randn((m,2560),device='cuda',dtype=torch.bfloat16)*0.1
        ids=torch.arange(2,device='cuda',dtype=torch.int32).expand(m,2).contiguous()
        weights=torch.softmax(torch.randn((m,2),device='cuda',dtype=torch.float32),dim=-1)
        dispatch=StandardDispatchOutput(x,None,StandardTopKOutput(weights,ids,None))
        actual=layer.quant_method.apply(layer,dispatch).hidden_states
        expected=torch.zeros_like(x,dtype=torch.float32)
        for e,r in enumerate(reference):
            gate=x @ r['gate_proj'].T
            up=x @ r['up_proj'].T
            intermediate=torch.nn.functional.silu(gate)*up
            down=intermediate @ r['down_proj'].T
            expected+=down.float()*weights[:,e,None]
        relative=((actual.float()-expected).norm()/expected.norm()).item()
        assert torch.isfinite(actual).all() and relative<0.02,relative
        print(f'Actual MTP W4A16 experts M={m}: relative L2={relative}',flush=True)
    from graph_probe import check_replay
    check_replay(layer, dispatch)
torch.distributed.destroy_process_group()
