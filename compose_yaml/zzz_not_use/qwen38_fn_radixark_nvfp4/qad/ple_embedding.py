"""TP1 packed NVFP4 PLE storage for the pinned SGLang file-offload API."""
import math
import os
import re
import torch
from torch import nn
from nvfp4_ple import gather
from sglang.srt.models.qwen4_exp_ple_table import (
    allocate_ple_host_table, make_ple_file_prefetcher, make_ple_file_rss_trimmer,
)


class NVFP4PLEEmbedding(nn.Module):
    def __init__(self, rows, dim, shards, directory):
        super().__init__()
        if dim % 16 or rows <= 0 or shards <= 0:
            raise ValueError('Invalid packed PLE geometry')
        self.org_vocab_size = rows
        self.embedding_dim = dim
        self.shards = shards
        self.shard_rows = math.ceil(rows / shards)
        self._disk = None
        if os.environ.get('SGLANG_QAD_PLE_IO_URING') == '1':
            from ple_disk import CheckpointRows
            from sglang.srt.server_args import get_global_server_args
            self._disk = CheckpointRows(rows, dim, shards, get_global_server_args().model_path)
        if self._disk is None:
            self.packed = allocate_ple_host_table((rows, dim//2), torch.uint8,
                        backend='file', table_dir=directory, tag='qad-nvfp4-packed')
            self.scales = allocate_ple_host_table((rows, dim//16), torch.float8_e4m3fn,
                        backend='file', table_dir=directory, tag='qad-nvfp4-scales')
        self.register_buffer('weight_scale', torch.ones(1,device='cuda',dtype=torch.bfloat16))
        self.register_buffer('weight_scale_2', torch.full((1,),float('nan'),device='cuda',dtype=torch.float32))
        tables = (self.packed, self.scales) if self._disk is None else ()
        self._prefetchers = [make_ple_file_prefetcher(t) for t in tables]
        self._trimmers = [make_ple_file_rss_trimmer(t) for t in tables]
        self._loaded = set()
        self._global_loaded = False
        self.ready = False

    def load_tensor(self, suffix, value):
        if suffix == 'weight_scale_2':
            if value.numel()!=1 or value.dtype!=torch.float32 or not torch.isfinite(value).all().item() or value.item()<=0:
                raise ValueError('PLE global scale must be finite positive FP32')
            if self._global_loaded:raise ValueError('Duplicate PLE global scale')
            self.weight_scale_2.copy_(value.reshape(1));self._global_loaded=True
            return True
        m = re.fullmatch(r'shard_(\d+)\.(weight|weight_scale)',suffix)
        if not m:return False
        index,kind=int(m[1]),m[2]
        if not 0<=index<self.shards or (index,kind) in self._loaded:
            raise ValueError('Invalid or duplicate PLE shard')
        lo=index*self.shard_rows;hi=min(lo+self.shard_rows,self.org_vocab_size)
        dtype = torch.uint8 if kind == 'weight' else torch.float8_e4m3fn
        width = self.embedding_dim // (2 if kind == 'weight' else 16)
        if value.dtype != dtype or tuple(value.shape) != (hi-lo, width):
            raise ValueError(f'PLE {suffix}: expected {(hi-lo,width)} {dtype}, got {value.shape} {value.dtype}')
        if self._disk is not None:
            self._disk.load(suffix, value)
        else:
            dest=self.packed if kind=='weight' else self.scales
            dest[lo:hi].copy_(value.to(device='cpu'))
        self._loaded.add((index,kind))
        return True

    def finish_load(self):
        expected={(i,k) for i in range(self.shards) for k in ('weight','weight_scale')}
        if self._loaded!=expected or not self._global_loaded:
            raise ValueError(f'Incomplete NVFP4 PLE: {len(self._loaded)}/{len(expected)} tensors, global={self._global_loaded}')
        if self._disk is not None:
            self._disk.finish()
        for tensor in ((self.packed,self.scales) if self._disk is None else ()):
            with open(tensor._sglang_ple_file_path,'rb') as f:
                os.fsync(f.fileno())
        self.ready=True
        backend='checkpoint-io_uring+uva' if self._disk is not None else 'file-copy-uva'
        print(f'QAD_NVFP4_PLE_READY rows={self.org_vocab_size} dim={self.embedding_dim} shards={self.shards} backend={backend}',flush=True)

    def allocate_output(self,shape,device):
        return torch.empty(shape,dtype=torch.bfloat16,device=device)

    def gather(self,ids,out=None):
        if not self.ready:raise RuntimeError('PLE used before complete verified load')
        if self._disk is not None:
            return self._disk.lookup(ids.contiguous().long(), self.weight_scale_2, out)
        flat=ids.reshape(-1).long()
        for p in self._prefetchers:
            if p is not None:p.enqueue(flat)
        return gather(self.packed,self.scales,self.weight_scale_2,ids.contiguous().long(),out=out)

    def reduce(self,output):return output
    def forward(self,ids):return self.gather(ids)
