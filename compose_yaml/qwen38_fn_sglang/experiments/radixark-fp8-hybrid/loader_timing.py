"""Interleaved H2D copy A/B with the same warmed checkpoint tensor and GPU."""
import json
import statistics
import time
from pathlib import Path
import torch
from safetensors import safe_open

p=Path('/hf/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4/layer-00000-experts-0000-0127.safetensors')
results=[]
with safe_open(p,framework='pt',device='cpu') as handle:
    for name in sorted(k for k in handle.keys() if k.endswith('.weight'))[:3]:
        source=handle.get_tensor(name)
        anonymous=source.clone()
        target=torch.empty_like(source,device='cuda')
        target.copy_(source);torch.cuda.synchronize()
        samples={'mapped':[],'clone':[]}
        for rep in range(20):
            for mode in (['mapped','clone'] if rep%2==0 else ['clone','mapped']):
                torch.cuda.synchronize();start=time.perf_counter()
                value=source if mode=='mapped' else source.clone()
                target.copy_(value);torch.cuda.synchronize()
                samples[mode].append((time.perf_counter()-start)*1000)
        assert torch.equal(target.cpu().reshape(-1).view(torch.uint8),source.reshape(-1).view(torch.uint8))
        results.append(dict(name=name,bytes=source.nbytes,samples_ms=samples,
                            medians_ms={k:statistics.median(v) for k,v in samples.items()}))
print('LOADER_COPY_AB '+json.dumps(results),flush=True)

# A contiguous copy is not the engine's destination layout. Measure the actual
# native FusedMoE weight loader too, including its slices/transposes/interleave.
from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
from sglang.srt.distributed import init_distributed_environment,initialize_model_parallel
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.utils import initialize_moe_config
root=p.parent
args=ServerArgs(model_path=str(root),quantization='modelopt_fp4',disable_cuda_graph=True,moe_runner_backend='flashinfer_cutlass')
set_global_server_args_for_scheduler(args);initialize_moe_config(args)
init_distributed_environment(1,0,'tcp://127.0.0.1:29585',0,'nccl');initialize_model_parallel(1)
config=ModelOptFp4Config.from_config(json.loads((root/'hf_quant_config.json').read_text()));config.packed_modules_mapping={}
with torch.device('cuda'):
    layer=FusedMoE(num_experts=2,hidden_size=2560,intermediate_size=640,layer_id=0,top_k=2,
                   params_dtype=torch.bfloat16,quant_config=config,prefix='model.layers.0.mlp.experts')
actual_results=[]
with safe_open(p,framework='pt',device='cpu') as handle:
    for projection,destination,part in [('gate_proj','w13_weight','w1'),('up_proj','w13_weight','w3'),('down_proj','w2_weight','w2')]:
        name=f'model.language_model.layers.0.mlp.experts.0.{projection}.weight'
        source=handle.get_tensor(name);param=getattr(layer,destination)
        samples={'mapped':[],'clone':[]}
        for rep in range(20):
            for mode in (['mapped','clone'] if rep%2==0 else ['clone','mapped']):
                torch.cuda.synchronize();start=time.perf_counter()
                value=source if mode=='mapped' else source.clone()
                param.weight_loader(param,value,destination,shard_id=part,expert_id=0)
                torch.cuda.synchronize();samples[mode].append((time.perf_counter()-start)*1000)
        actual_results.append(dict(name=name,samples_ms=samples,medians_ms={k:statistics.median(v) for k,v in samples.items()}))
print('LOADER_FUSED_AB '+json.dumps(actual_results),flush=True)

first_use=[]
with safe_open(p,framework='pt',device='cpu') as handle:
    for projection,destination,part in [('gate_proj','w13_weight','w1'),('up_proj','w13_weight','w3'),('down_proj','w2_weight','w2')]:
        param=getattr(layer,destination);samples={'mapped':[],'clone':[]}
        for expert in range(8,68):
            source=handle.get_tensor(f'model.language_model.layers.0.mlp.experts.{expert}.{projection}.weight')
            # Each source view is copied just once; the earlier warm-copy test
            # intentionally measures a different, repeatedly reused mapping.
            mode='mapped' if expert%2==0 else 'clone'
            torch.cuda.synchronize();start=time.perf_counter()
            value=source if mode=='mapped' else source.clone()
            param.weight_loader(param,value,destination,shard_id=part,expert_id=0)
            torch.cuda.synchronize();samples[mode].append((time.perf_counter()-start)*1000)
        first_use.append(dict(projection=projection,samples_ms=samples,medians_ms={k:statistics.median(v) for k,v in samples.items()}))
print('LOADER_FIRST_USE_AB '+json.dumps(first_use),flush=True)
torch.distributed.destroy_process_group()
