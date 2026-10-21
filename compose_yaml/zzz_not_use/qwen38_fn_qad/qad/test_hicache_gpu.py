"""Tiny native CUDA-pool round trip; does not load or replace any model."""
from types import SimpleNamespace as NS
import torch
import tempfile, json
from pathlib import Path
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from qad_hicache import CompleteHostPool, ExtraTensor, ple_tensors

with tempfile.TemporaryDirectory() as config_dir:
    Path(config_dir, 'config.json').write_text(json.dumps(dict(model_type='llama', architectures=['LlamaForCausalLM'], hidden_size=128, num_attention_heads=1, num_key_value_heads=1, num_hidden_layers=2, intermediate_size=256, vocab_size=128)))
    set_global_server_args_for_scheduler(ServerArgs(model_path=config_dir, device='cuda'))
# Test-only: native HiCache reserves 10 GB even for sub-MB fixtures. Keep
# production reserve unchanged; this isolated process allocates only tiny pools.
import sglang.srt.mem_cache.pool_host.base as host_base
import sglang.srt.mem_cache.memory_pool_host as host_mamba
host_base.HICACHE_HOST_MEMORY_RESERVE_BYTES = 0
host_mamba.HICACHE_HOST_MEMORY_RESERVE_BYTES = 0
torch.manual_seed(38121)
kv=MHATokenToKVPool(size=128,page_size=64,dtype=torch.float8_e4m3fn,
    head_num=1,head_dim=128,layer_num=2,device='cuda',enable_memory_saver=False)
for b in (*kv.k_buffer,*kv.v_buffer):b.copy_(torch.randn(b.shape,device='cuda').to(b.dtype))
base=MHATokenToKVPoolHost(kv,2,0,64,'page_first')
qsa=[ExtraTensor(f'qsa-{i}',torch.randn(48,1,128,device='cuda',dtype=torch.bfloat16),i) for i in range(2)]
host=CompleteHostPool(base,qsa,compression=4)
hi=torch.arange(64,128,dtype=torch.int64,device='cuda')
di=torch.arange(0,64,dtype=torch.int64,device='cuda')
expected=[b[:64].clone() for b in (*kv.k_buffer,*kv.v_buffer)]
expected_qsa=[e.tensor[:16].clone() for e in qsa]
host.backup_from_device_all_layer(kv,hi.cpu(),di,'kernel')
torch.cuda.synchronize()
page=host.get_data_page(64)
host.set_from_flat_data_page(128,page)
for b in (*kv.k_buffer,*kv.v_buffer):b.zero_()
for e in qsa:e.tensor.zero_()
for i in range(2):host.load_to_device_per_layer(kv,hi+64,di+64,i)
torch.cuda.synchronize()
for b,expected_b in zip((*kv.k_buffer,*kv.v_buffer),expected):
    assert torch.equal(b[64:128].view(torch.uint8),expected_b.view(torch.uint8))
for e,expected_e in zip(qsa,expected_qsa):torch.testing.assert_close(e.tensor[16:32],expected_e,rtol=0,atol=0)
host.destroy()

# Native Mamba host, including PLE views. Tiny synthetic device state has the
# same axes used by MambaPool; exercise actual pinned-memory transfer kernels.
state=NS(conv=[torch.randn(2,9,8,3,device='cuda',dtype=torch.bfloat16)],
         temporal=torch.randn(2,9,2,8,8,device='cuda',dtype=torch.float32))
conv=torch.randn(1,9,8,3,device='cuda',dtype=torch.bfloat16)
context=torch.arange(27,device='cuda',dtype=torch.int64).reshape(9,3)+2**54
mp=NS(num_mamba_layers=2,mamba_cache=state,device='cuda',size=8,
      _slot_siblings=[NS(conv_state=conv),NS(context=context)])
mb=CompleteHostPool(MambaPoolHost(mp,2,0,layout='page_first'),ple_tensors(mp),label='ple')
hi=torch.tensor([3,5],device='cuda');di=torch.tensor([1,2],device='cuda')
original=[state.conv[0][:,di].clone(),state.temporal[:,di].clone(),conv[:,di].clone(),context[di].clone()]
mb.backup_from_device_all_layer(mp,hi.cpu(),di,'kernel');torch.cuda.synchronize()
pages=[mb.get_data_page(i) for i in (3,5)]
for i,p in zip((7,9),pages):mb.set_from_flat_data_page(i,p)
for t in (state.conv[0],state.temporal,conv,context):t.zero_()
for i in range(2):mb.load_to_device_per_layer(mp,torch.tensor([7,9],device='cuda'),torch.tensor([4,6],device='cuda'),i)
torch.cuda.synchronize()
for a,b in zip((state.conv[0][:,[4,6]],state.temporal[:,[4,6]],conv[:,[4,6]],context[[4,6]]),original):
    torch.testing.assert_close(a,b,rtol=0,atol=0)
print('Native FP8 KV + BF16 QSA + Mamba + PLE CUDA round-trip: PASS')
print('Peak allocated CUDA bytes:',torch.cuda.max_memory_allocated())
