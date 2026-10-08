import tempfile
import os
import torch
from nvfp4_ple import gather

# Independent PyTorch lookup reference exercises all 16 codes, signs, FP8
# block scales, the global scale, row selection and out-of-partition IDs.
torch.manual_seed(18)
for device in ('cuda', 'cpu', 'file'):
    packed=torch.arange(256,dtype=torch.int32).repeat(5)[:8*80].to(torch.uint8).reshape(8,80)
    scales=(torch.rand(8,10)*4).to(torch.float8_e4m3fn)
    w=packed.to(device if device != "file" else "cpu");s=scales.to(device if device != "file" else "cpu")
    if device=="file":
        from sglang.srt.models.qwen4_exp_ple_table import allocate_ple_host_table
        directory=tempfile.TemporaryDirectory()
        w=allocate_ple_host_table(shape=packed.shape,dtype=torch.uint8,backend="file",table_dir=directory.name,tag="packed")
        s=allocate_ple_host_table(shape=scales.shape,dtype=torch.float8_e4m3fn,backend="file",table_dir=directory.name,tag="scales")
        w.copy_(packed);s.copy_(scales)
    if device=='cpu':w=w.pin_memory();s=s.pin_memory()
    scale=torch.tensor([0.01372549],device='cuda',dtype=torch.float32)
    ids=torch.tensor([4,11,3,12,7,4],device='cuda',dtype=torch.int64)
    codes=torch.stack((packed & 15,packed >> 4),dim=-1).reshape(8,160).long()
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
    decoded=(lut[codes]*(scales.float().repeat_interleave(16,dim=1)*scale.cpu())).bfloat16()
    expected=torch.stack([decoded[i-4] if 4<=i<12 else torch.zeros(160,dtype=torch.bfloat16) for i in ids.tolist()]).cuda()
    if device=='file':
        # Exercise real pageable-file faults, not only already-resident pages.
        from sglang.srt.models.qwen4_exp_ple_table import _madvise, _MADV_DONTNEED
        for table in (w,s):
            with open(table._sglang_ple_file_path,'rb') as f:
                os.fsync(f.fileno())
                assert _madvise(table.data_ptr(),table.numel()*table.element_size(),_MADV_DONTNEED)
                os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    actual=gather(w,s,scale,ids,start=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    # Captured row gather is required by SGLang decode/MTP execution.
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):gather(w,s,scale,ids,start=4)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    out=torch.empty_like(actual)
    with torch.cuda.graph(graph):gather(w,s,scale,ids,start=4,out=out)
    graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(out,expected,rtol=0,atol=0)
    print(device, 'exact BF16 output and CUDA graph replay passed',flush=True)
