"""Checkpoint byte offsets, tail shards, eager I/O and changing graph IDs."""
import json
import torch
import pytest
from safetensors import safe_open
from safetensors.torch import save_file
from ple_disk import CheckpointRows, CheckpointRssTrimmer


def test_mmap_trimmer_preserves_file_and_rejects_anonymous_storage(tmp_path):
    file=tmp_path/'immutable.safetensors'
    expected=torch.arange(40000,dtype=torch.int32)
    save_file({'values':expected},str(file))
    with safe_open(file,framework='pt',device='cpu') as reader:
        mapped=reader.get_tensor('values')
    trimmer=CheckpointRssTrimmer([mapped],[file],0)
    assert trimmer.resident_bytes()>=0
    trimmer.trim_once(force=True)
    torch.testing.assert_close(mapped,expected,rtol=0,atol=0)
    with pytest.raises(ValueError,match='file mapping'):
        CheckpointRssTrimmer([expected],[file],0)
    trimmer.close()


def test_checkpoint_rows_exact_and_graph(tmp_path):
    torch.manual_seed(31)
    rows, dim, shards = 101, 160, 3
    packed = torch.randint(0, 256, (rows, dim//2), dtype=torch.uint8)
    scales = (torch.rand(rows, dim//16) * 3).to(torch.float8_e4m3fn)
    prefix = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.'
    index = {}
    for i in range(shards):
        lo, hi = i*34, min((i+1)*34, rows)
        tensors = {prefix+f'shard_{i}.weight': packed[lo:hi].clone(),
                   prefix+f'shard_{i}.weight_scale': scales[lo:hi].clone()}
        file = f'part-{i}.safetensors'
        save_file(tensors, str(tmp_path/file))
        index.update({name: file for name in tensors})
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
    table = CheckpointRows(rows, dim, shards, tmp_path)
    for name, file in index.items():
        with safe_open(tmp_path/file, framework='pt', device='cpu') as source:
            table.load(name.split(prefix)[1], source.get_tensor(name))
    table.finish()
    factor = torch.tensor([.01372549], device='cuda')
    lut = torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(rows, dim).long()
    decoded = (lut[codes] * (scales.float().repeat_interleave(16, 1) * factor.cpu())).bfloat16().cuda()
    def reference(ids):
        valid = (ids >= 0) & (ids < rows)
        return torch.where(valid[..., None], decoded[ids.clamp(0, rows-1)], 0)
    # All file extents, duplicate rows, boundaries and invalid IDs are used.
    ids = torch.arange(-1, 103, device='cuda').repeat(4).reshape(26, 16)
    actual = table.lookup(ids, factor)
    torch.testing.assert_close(actual, reference(ids), rtol=0, atol=0)
    assert table.cache.stats()['cache_bytes'] < 128*1024
    assert table.cache.stats()['submit_calls'] > 0
    ids = torch.tensor([[0,33,34,67,68,100,-1,101]], device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): table.lookup(ids, factor)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = table.lookup(ids, factor)
    ids.copy_(torch.tensor([[100,68,67,34,33,0,101,-1]], device='cuda'))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, reference(ids), rtol=0, atol=0)
    # Clean checkpoint mapping pages can be dropped during graph replay;
    # GB10 must fault them back correctly without rewriting the checkpoint.
    for _ in range(8):
        graph.replay()
        table.trimmer.trim_once(force=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, reference(ids), rtol=0, atol=0)
    # Reuse and grow staging while preserving invalid-ID zeros.
    for count in (300, 1000, 400):
        ids = (torch.arange(count, device='cuda') % 103) - 1
        torch.testing.assert_close(table.lookup(ids, factor), reference(ids), rtol=0, atol=0)
    table.trimmer.close()
