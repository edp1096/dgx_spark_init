import json,pathlib,sys,tempfile
import torch
from ple_embedding import NVFP4PLEEmbedding
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
root=pathlib.Path(sys.argv[1])
w=torch.frombuffer(bytearray((root/'weights.bin').read_bytes()),dtype=torch.uint8).clone().reshape(4,80)
s=torch.frombuffer(bytearray((root/'scales.bin').read_bytes()),dtype=torch.float8_e4m3fn).clone().reshape(4,10)
g=torch.frombuffer(bytearray((root/'global.bin').read_bytes()),dtype=torch.float32).clone()
reference=NVFP4QTensor(torch.Size((4,160)),torch.bfloat16,w).dequantize(scale=s,double_scale=g,block_sizes={-1:16})
with tempfile.TemporaryDirectory() as directory:
    emb=NVFP4PLEEmbedding(4,160,2,directory)
    # Test out-of-order arrival of weights/scales/global scalar.
    emb.load_tensor('shard_1.weight_scale',s[2:]);emb.load_tensor('shard_0.weight',w[:2])
    try:emb.finish_load()
    except ValueError:pass
    else:raise AssertionError('Incomplete load accepted')
    emb.load_tensor('weight_scale_2',g)
    emb.load_tensor('shard_1.weight',w[2:]);emb.load_tensor('shard_0.weight_scale',s[:2])
    try:emb.load_tensor('shard_0.weight',w[:2])
    except ValueError:pass
    else:raise AssertionError('Duplicate tensor accepted')
    emb.finish_load()
    ids=torch.tensor([3,0,2,1,3],dtype=torch.int64,device='cuda')
    actual=emb(ids)
    torch.testing.assert_close(actual.cpu(),reference[ids.cpu()],rtol=0,atol=0)
    torch.cuda.synchronize()
    print('Actual checkpoint PLE rows match NVIDIA ModelOpt dequantization exactly; missing/duplicate load checks passed',flush=True)

# Broader arithmetic check across all finite nonnegative E4M3 scale codes.
from nvfp4_ple import gather
torch.manual_seed(19)
w=torch.randint(0,256,(4096,80),dtype=torch.uint8)
s=torch.randint(0,127,(4096,10),dtype=torch.uint8).view(torch.float8_e4m3fn)
ids=torch.arange(4096,device='cuda',dtype=torch.int64)
for factor in (1e-8,0.01372549,0.37,1.0,16.3):
    g=torch.tensor([factor],dtype=torch.float32)
    reference=NVFP4QTensor(torch.Size((4096,160)),torch.bfloat16,w).dequantize(
        scale=s,double_scale=g,block_sizes={-1:16})
    actual=gather(w.cuda(),s.cuda(),g.cuda(),ids)
    torch.testing.assert_close(actual.cpu(),reference,rtol=0,atol=0)
print('4096 randomized rows across five global scales match ModelOpt exactly',flush=True)
