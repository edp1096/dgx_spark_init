"""Round-trip tests for QAD payloads; no model weights or GPU needed."""
import os
import tempfile
import unittest
from types import SimpleNamespace as NS

import torch
from qad_hicache import CompleteHostPool, ExtraTensor, ple_tensors, qsa_tensors, budgets, prepare, ENVELOPE_BYTES


class TinyHost:
    """Small native-pool stand-in; tests target dependent-state transport."""
    def __init__(self, page_size=4, size=32, layers=2):
        self.page_size, self.size, self.pin_memory = page_size, size, False
        self.layer_num = layers
        self.values = torch.zeros(layers, size, 2, dtype=torch.float32)
        self.device = 'cpu'
        self.logical_size = size
        self.layout = 'page_first'
    def get_data_page(self, index, flat=True):
        return self.values[:, index:index+self.page_size].contiguous().flatten()
    def get_dummy_flat_data_page(self):
        return torch.empty(self.layer_num*self.page_size*2, dtype=torch.float32)
    def set_from_flat_data_page(self, index, data):
        self.values[:, index:index+self.page_size].copy_(data.reshape(self.layer_num,self.page_size,2))
    def backup_from_device_all_layer(self, pool, hi, di, backend):
        self.values[:, hi] = pool.values[:, di]
    def load_to_device_per_layer(self, pool, hi, di, layer, backend, is_draft=False):
        pool.values[0 if is_draft else layer, di] = self.values[layer, hi]
    def destroy(self): pass


def fixture():
    qsa=[ExtraTensor('qsa-target',torch.arange(32,dtype=torch.bfloat16).reshape(16,2),0),
         ExtraTensor('qsa-draft',torch.arange(100,132,dtype=torch.bfloat16).reshape(16,2),1)]
    return CompleteHostPool(TinyHost(),qsa,compression=2),qsa


class PayloadTests(unittest.TestCase):
    def test_qsa_relocated_pages_and_draft(self):
        host,extras=fixture()
        device=TinyHost();device.values.copy_(torch.arange(128).reshape(2,32,2))
        hi=torch.tensor([4,5,6,7,12,13,14,15]);di=torch.tensor([16,17,18,19,8,9,10,11])
        kv=device.values[:,di].clone()
        expected=[e.tensor[di[::2]//2].clone() for e in extras]
        host.backup_from_device_all_layer(device,hi,di,'kernel')
        pages=[host.get_data_page(i).clone() for i in (4,12)]
        fresh,newextras=fixture()
        for e in newextras:e.tensor.fill_(-1)
        for offset,page in zip((0,20),pages):fresh.set_from_flat_data_page(offset,page)
        loadhi=torch.tensor([0,1,2,3,20,21,22,23]);loaddi=torch.tensor([24,25,26,27,4,5,6,7])
        target=TinyHost();draft=TinyHost(layers=1)
        fresh.load_to_device_per_layer(target,loadhi,loaddi,0)
        fresh.load_to_device_per_layer(draft,loadhi,loaddi,1,is_draft=True)
        torch.testing.assert_close(target.values[0,loaddi],kv[0],rtol=0,atol=0)
        torch.testing.assert_close(draft.values[0,loaddi],kv[1],rtol=0,atol=0)
        for extra,value in zip(newextras,expected):
            torch.testing.assert_close(extra.tensor[loaddi[::2]//2],value,rtol=0,atol=0)

    def test_ple_views_and_int64_no_precision_loss(self):
        conv=torch.arange(96,dtype=torch.bfloat16).reshape(2,8,6)
        context=torch.arange(24,dtype=torch.int64).reshape(8,3)+2**54
        pool=NS(_slot_siblings=[NS(conv_state=conv),NS(context=context)])
        host=CompleteHostPool(TinyHost(page_size=1,size=8),ple_tensors(pool),label='ple')
        device=TinyHost(page_size=1,size=8)
        host.backup_from_device_all_layer(device,torch.tensor([2]),torch.tensor([5]),'kernel')
        payload=host.get_data_page(2)
        saved_conv,saved_context=conv[:,5].clone(),context[5].clone()
        conv.zero_();context.zero_()
        host.set_from_flat_data_page(3,payload)
        host.load_to_device_per_layer(device,torch.tensor([3]),torch.tensor([7]),0)
        torch.testing.assert_close(conv[:,7],saved_conv,rtol=0,atol=0)
        self.assertTrue(torch.equal(context[7],saved_context))
        self.assertTrue(torch.equal(context[5],torch.zeros(3,dtype=torch.int64)))

    def test_reject_incomplete_or_misaligned_pages(self):
        host,_=fixture();dev=TinyHost()
        for ids in ([1,2,3,4],[0,1],[0,2,1,3]):
            with self.assertRaises(ValueError):host.backup_from_device_all_layer(dev,torch.tensor(ids),torch.tensor(ids),'kernel')

    def test_invalid_payload_does_not_mutate(self):
        host,_=fixture()
        host.base.values.fill_(9)
        page=host.get_data_page(0)
        for data in (page[:-1],torch.cat([page,torch.zeros(1,dtype=torch.uint8)]),page.clone()):
            if data.numel()==page.numel():data[0]^=1
            with self.assertRaises(ValueError):host.set_from_flat_data_page(0,data)
            self.assertTrue(torch.all(host.base.values==9))

    def test_same_size_corruption_does_not_mutate(self):
        host,_=fixture()
        host.base.values.fill_(9)
        for buffer in host.buffers: buffer.fill_(7)
        original = host.get_data_page(0)
        for offset in (40, original.numel()//2, original.numel()-1):
            page = original.clone(); page[offset] ^= 1
            with self.assertRaises(ValueError): host.set_from_flat_data_page(0,page)
            self.assertTrue(torch.all(host.base.values==9))
            self.assertTrue(all(torch.all(buffer==7) for buffer in host.buffers))

    def test_file_corruption_is_miss_and_rewritable(self):
        from pathlib import Path
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig, PoolName
        cfg=HiCacheStorageConfig(tp_rank=0,tp_size=1,pp_rank=0,pp_size=1,attn_cp_rank=0,attn_cp_size=1,
            is_mla_model=False,enable_storage_metrics=False,is_page_first_layout=True,model_name='qad-integrity',
            extra_config={'enable_metadata_cache':True})
        with tempfile.TemporaryDirectory() as directory:
            backend=HiCacheFile(cfg,file_path=directory)
            host,_=fixture()
            host.base.values.fill_(9)
            for buffer in host.buffers: buffer.fill_(7)
            original=host.get_data_page(0)
            backend.register_mem_host_pool_v2(host,PoolName.KV)
            for mode in ('truncated','overlong','header','body','checksum','missing'):
                self.assertTrue(backend._write_page(PoolName.KV,'key',host,0))
                path=Path(directory)/(backend._get_suffixed_key('key')+'.bin')
                raw=bytearray(path.read_bytes())
                if mode=='truncated': raw=raw[:-1]
                elif mode=='overlong': raw+=b'!'
                elif mode=='header': raw[0]^=1
                elif mode=='body': raw[44]^=1
                elif mode=='checksum': raw[-1]^=1
                if mode=='missing': path.unlink()
                else: path.write_bytes(raw)
                self.assertFalse(backend._read_page(PoolName.KV,'key',host,0),mode)
                self.assertTrue(torch.equal(host.get_data_page(0),original))
                self.assertFalse(path.exists())
                self.assertTrue(backend._write_page(PoolName.KV,'key',host,0))
                self.assertTrue(backend._read_page(PoolName.KV,'key',host,0))
                self.assertTrue(torch.equal(host.get_data_page(0),original))

    def test_file_eviction_and_interrupted_atomic_write(self):
        from pathlib import Path
        from unittest.mock import patch
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig, PoolName
        host,_=fixture()
        for buffer in host.buffers: buffer.fill_(7)
        cfg=HiCacheStorageConfig(tp_rank=0,tp_size=1,pp_rank=0,pp_size=1,attn_cp_rank=0,attn_cp_size=1,
            is_mla_model=False,enable_storage_metrics=False,is_page_first_layout=True,model_name='qad-eviction',
            extra_config={'enable_metadata_cache':True,'max_size':host.page_bytes*2,'eviction_ratio':1.0,'min_free_space':0})
        with tempfile.TemporaryDirectory() as directory:
            backend=HiCacheFile(cfg,file_path=directory)
            backend.register_mem_host_pool_v2(host,PoolName.KV)
            for key in ('a','b'): self.assertTrue(backend._write_page(PoolName.KV,key,host,0))
            self.assertTrue(backend._read_page(PoolName.KV,'a',host,0))
            self.assertTrue(backend._write_page(PoolName.KV,'c',host,0))
            self.assertTrue(backend.exists('a'))
            self.assertFalse(backend.exists('b'))
            self.assertTrue(backend.exists('c'))
            with patch('os.replace',side_effect=OSError('interrupted before atomic rename')):
                self.assertFalse(backend._write_page(PoolName.KV,'d',host,0))
            self.assertFalse(backend.exists('d'))
            self.assertEqual(list(Path(directory).glob('*.tmp.*')),[])
            # A killed writer can leave a temp file. A new backend must never
            # advertise or read it as a complete checkpoint.
            partial=Path(directory)/(backend._get_suffixed_key('partial')+'.bin.tmp.killed')
            partial.write_bytes(b'partial')
            fresh=HiCacheFile(cfg,file_path=directory)
            self.assertFalse(fresh.exists('partial'))
            self.assertTrue(fresh._write_page(PoolName.KV,'d',host,0))
            self.assertTrue(fresh._read_page(PoolName.KV,'d',host,0))
            self.assertLessEqual(sum(f.stat().st_size for f in Path(directory).glob('*.bin')),host.page_bytes*2)

    def test_generic_kv_io_rejects_corruption_without_worker_exception(self):
        from pathlib import Path
        from sglang.srt.managers.cache_controller import HiCacheController
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig, PoolName
        host,_=fixture()
        for buffer in host.buffers: buffer.fill_(7)
        cfg=HiCacheStorageConfig(tp_rank=0,tp_size=1,pp_rank=0,pp_size=1,attn_cp_rank=0,attn_cp_size=1,
            is_mla_model=False,enable_storage_metrics=False,is_page_first_layout=True,model_name='qad-generic',extra_config={})
        with tempfile.TemporaryDirectory() as directory:
            backend=HiCacheFile(cfg,file_path=directory)
            backend.register_mem_host_pool_v2(host,PoolName.KV)
            ctl=NS(mem_pool_host=NS(anchor_entry=NS(host_pool=host)),storage_backend=backend,page_size=4)
            for key in ('a','b'): self.assertTrue(backend._write_page(PoolName.KV,key,host,0))
            path=Path(directory)/(backend._get_suffixed_key('b')+'.bin')
            raw=bytearray(path.read_bytes());raw[44]^=1;path.write_bytes(raw)
            completed=[]
            op=NS(increment=lambda n:(completed.append(n) or True))
            HiCacheController._generic_page_get(ctl,op,['a','b'],torch.arange(8))
            self.assertEqual(completed,[4])
            self.assertFalse(path.exists())
            self.assertTrue(backend._write_page(PoolName.KV,'b',host,0))
            HiCacheController._generic_page_get(ctl,op,['b'],torch.arange(4))
            self.assertEqual(completed,[4,4])

    def test_killed_file_writer_preserves_committed_pages(self):
        import subprocess, sys, json, selectors
        from pathlib import Path
        from dataclasses import asdict
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig, PoolName
        host,_=fixture()
        for buffer in host.buffers: buffer.fill_(7)
        cfg=HiCacheStorageConfig(tp_rank=0,tp_size=1,pp_rank=0,pp_size=1,attn_cp_rank=0,attn_cp_size=1,
            is_mla_model=False,enable_storage_metrics=False,is_page_first_layout=True,model_name='qad-killed-writer',extra_config={})
        code = """import os,sys,json
from test_hicache import fixture
from sglang.srt.mem_cache.hicache_storage import HiCacheFile,HiCacheStorageConfig,PoolName
host,_=fixture()
for buffer in host.buffers: buffer.fill_(7)
backend=HiCacheFile(HiCacheStorageConfig(**json.loads(sys.argv[1])),file_path=sys.argv[2])
def pause_before_publish(src,dst):
 print('BEFORE_RENAME',flush=True)
 sys.stdin.read()
 raise RuntimeError('parent should kill this child')
os.replace=pause_before_publish
backend._write_page(PoolName.KV,'interrupted',host,0)
"""
        with tempfile.TemporaryDirectory() as directory:
            backend=HiCacheFile(cfg,file_path=directory)
            self.assertTrue(backend._write_page(PoolName.KV,'committed',host,0))
            proc=subprocess.Popen([sys.executable,'-c',code,json.dumps(asdict(cfg)),directory],
                                  stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stdout,selectors.EVENT_READ)
                    self.assertTrue(selector.select(timeout=60),'child did not reach atomic rename')
                    self.assertEqual(proc.stdout.readline().strip(),'BEFORE_RENAME')
                proc.kill();proc.wait(timeout=10)
            finally:
                if proc.poll() is None:proc.kill();proc.wait(timeout=10)
                proc.stdin.close();proc.stdout.close();proc.stderr.close()
            self.assertTrue(list(Path(directory).glob('*.tmp.*')))
            fresh=HiCacheFile(cfg,file_path=directory)
            self.assertTrue(fresh._read_page(PoolName.KV,'committed',host,0))
            self.assertFalse(fresh.exists('interrupted'))
            self.assertTrue(fresh._write_page(PoolName.KV,'interrupted',host,0))
            self.assertTrue(fresh._read_page(PoolName.KV,'interrupted',host,0))

    def test_unschedulable_request_is_rejected_after_output_clipping(self):
        from unittest.mock import patch
        from qad_hicache import admission_error
        args=NS(enable_hierarchical_cache=True,page_size=64)
        with patch.dict(os.environ,{'SGLANG_QAD_HICACHE':'1','SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION':'4096'}):
            # The native normalizer clips 32 requested output tokens to zero at
            # this boundary. The extra page still makes admission impossible.
            self.assertIn('at most 1048448 tokens',admission_error(args,1048512,0,1048576))
            self.assertIsNotNone(admission_error(args,1048480,32,1048576))
            self.assertIsNone(admission_error(args,1048479,32,1048576))
            self.assertIsNone(admission_error(args,1048000,32,1048576))
            self.assertIsNone(admission_error(args,1000000,20000,1048576))
            # Uses the actual pool capacity, not merely configured context size.
            self.assertIsNotNone(admission_error(args,4096,32,4096))
            args.enable_hierarchical_cache=False
            self.assertIsNone(admission_error(args,1048512,0,1048576))
        args.enable_hierarchical_cache=True
        with patch.dict(os.environ,{'SGLANG_QAD_HICACHE':'0'}):
            self.assertIsNone(admission_error(args,1048512,0,1048576))

    def test_disabled_state_and_unknown_sibling(self):
        self.assertEqual(ple_tensors(NS(_slot_siblings=[NS(conv_state=None),NS(context=None)])),[])
        with self.assertRaises(ValueError):ple_tensors(NS(_slot_siblings=[object()]))

    def test_budget_includes_extra_state_and_draft(self):
        kv=NS(size=1024,k_buffer=[torch.empty(1024,1,8)],v_buffer=[torch.empty(1024,1,8)])
        state=NS(conv=[torch.empty(2,9,8,3)],temporal=torch.empty(2,9,8,8))
        mp=NS(size=8,mamba_cache=state)
        extra=[ExtraTensor('qsa',torch.empty(256,1,8))]
        ple=[ExtraTensor('ple',torch.empty(9,24))]
        kg,mg=budgets((extra,ple,4),kv,mp,0.01,64,[NS(full_kv_pool=kv)])
        kv_row=128
        mamba_row=(2*8*3+2*8*8)*4
        slots=int(kg*1e9/kv_row)//64*64+64
        mslots=int(mg*1e9/mamba_row)+1
        used=slots*(kv_row+8+ENVELOPE_BYTES/64)+mslots*(mamba_row+96+ENVELOPE_BYTES)
        self.assertLessEqual(used,0.01*1e9)
        with self.assertRaises(ValueError):budgets((extra,ple,4),kv,mp,1e-12,64,[])

    def test_large_mamba_states_keep_two_host_slots(self):
        # Real QAD states are around 110 MiB each; proportional splitting used
        # to leave only one slot behind a 1 GB cache. Meta tensors use no RAM.
        kv=NS(size=1048576,k_buffer=[torch.empty(1048640,2,128,device='meta',dtype=torch.uint8)]*12,
              v_buffer=[torch.empty(1048640,2,128,device='meta',dtype=torch.uint8)]*12)
        mp=NS(size=8,mamba_cache=NS(conv=[torch.empty(1,9,8,device='meta')],
                                  temporal=torch.empty(1,9,28_000_000,device='meta')))
        qsa=[ExtraTensor('qsa',torch.empty(262160,1,128,device='meta',dtype=torch.bfloat16))]*13
        kg,mg=budgets((qsa,[],4),kv,mp,1,64,[])
        row=(8+28_000_000)*4
        self.assertGreaterEqual(int(mg*1e9/row)+1,2)
        self.assertGreater(kg,0)

    def test_opt_in_and_backend_guards(self):
        mp=NS(_slot_siblings=[NS(context=torch.zeros(9,3,dtype=torch.int64))])
        params=NS(req_to_token_pool=NS(mamba_pool=mp,mamba_map={0:0}),pp_size=1,attn_cp_size=1,mtp_draft_device_pools=(),page_size=64)
        args=NS(hicache_mem_layout='page_first',tp_size=1,hicache_size=1)
        from unittest.mock import patch
        with patch.dict(os.environ,{'SGLANG_QAD_HICACHE':'0'}):
            with self.assertRaises(ValueError):prepare(NS(),params,args,'file')
        with patch.dict(os.environ,{'SGLANG_QAD_HICACHE':'1'}):
            with self.assertRaises(ValueError):prepare(NS(),params,args,'mooncake')
            plan=prepare(NS(),params,args,'file')
            self.assertEqual(len(plan[1]),1)
            args.tp_size=2
            with self.assertRaises(ValueError):prepare(NS(),params,args,'file')

    def test_mtp_shared_ple_is_saved_once(self):
        from unittest.mock import patch
        mp=NS(_slot_siblings=[NS(context=torch.zeros(9,3,dtype=torch.int64))])
        params=NS(req_to_token_pool=NS(mamba_pool=mp,mamba_map={0:0}),pp_size=1,attn_cp_size=1,page_size=64)
        pool=NS(full_kv_pool=NS(layer_num=2),qsa_compress_ratio=4,qsa_compressed_k_buffer_pool=[torch.zeros(16,2)]*2)
        draft=NS(mamba_pool=mp,qsa_compress_ratio=4,qsa_compressed_k_buffer_pool=[torch.zeros(16,2)])
        params.mtp_draft_device_pools=(draft,)
        args=NS(hicache_mem_layout='page_first',tp_size=1,hicache_size=1)
        with patch.dict(os.environ,{'SGLANG_QAD_HICACHE':'1'}):
            plan=prepare(pool,params,args,'file')
            self.assertEqual(len(plan[0]),3)
            self.assertEqual(len(plan[1]),1)
            draft.mamba_pool=NS(_slot_siblings=mp._slot_siblings)
            with self.assertRaises(ValueError):prepare(pool,params,args,'file')

    def test_patched_native_strategy_wires_both_payloads(self):
        from unittest.mock import patch
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as asm
        mp=NS(size=8,mamba_cache=NS(conv=[torch.empty(1,9,2)],temporal=torch.empty(1,9,2)),
              _slot_siblings=[NS(context=torch.zeros(9,3,dtype=torch.int64))],get_kv_size_bytes=lambda:144)
        kv=NS(size=32,layer_num=2,k_buffer=[torch.empty(32,2)]*2,v_buffer=[torch.empty(32,2)]*2,get_kv_size_bytes=lambda:1024)
        pool=NS(full_kv_pool=kv,qsa_compress_ratio=2,qsa_compressed_k_buffer_pool=[torch.zeros(16,2)]*2,
                full_attention_layer_id_mapping={1:0,3:1},use_mla=False)
        alloc=NS(alloc=lambda n:None,free=lambda indices:None)
        params=NS(req_to_token_pool=NS(mamba_pool=mp,mamba_map={0:0,2:1},mamba_allocator=alloc),
                  pp_size=1,attn_cp_size=1,mtp_draft_device_pools=(),page_size=4,
                  token_to_kv_pool_allocator=None,tp_cache_group=None,attn_cp_cache_group=None,
                  attn_tp_cache_group=None,pp_cache_group=None)
        args=NS(hicache_mem_layout='page_first',tp_size=1,hicache_size=0.01,hicache_write_policy='write_through',hicache_io_backend='kernel')
        with patch.dict(os.environ,{'SGLANG_QAD_HICACHE':'1'}), \
             patch.object(asm,'build_kv_host_pool',return_value=TinyHost()), \
             patch.object(asm,'MambaPoolHost',return_value=TinyHost(page_size=1,size=8)), \
             patch.object(asm,'_get_allocator_type',return_value='default'), \
             patch.object(asm,'get_memory',return_value=NS(hicache_ratio=2)), \
             patch.object(asm,'HybridCacheController',side_effect=lambda *a,**k:NS(kwargs=k)):
            result=asm._MambaStrategy().build(cache=NS(),kvcache=pool,params=params,server_args=args,
                                             load_cache_event=None,storage_backend='file',model_name='qad')
            self.assertTrue(all(isinstance(e.host_pool,CompleteHostPool) for e in result.host_pool_group.entries))
            self.assertEqual(result.cache_controller.kwargs['model_name'],'qad-qad-complete-v2')

    def test_real_file_backend_fresh_reader(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig, PoolName, PoolTransfer
        cfg=HiCacheStorageConfig(tp_rank=0,tp_size=1,pp_rank=0,pp_size=1,attn_cp_rank=0,attn_cp_size=1,
            is_mla_model=False,enable_storage_metrics=False,is_page_first_layout=True,model_name='qad-test-v1',
            extra_config={'enable_metadata_cache':False})
        with tempfile.TemporaryDirectory() as directory:
            os.environ['SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR']=directory
            host,_=fixture();dev=TinyHost()
            ids=torch.tensor([4,5,6,7])
            host.backup_from_device_all_layer(dev,ids,ids,'kernel')
            expected=host.get_data_page(4).clone()
            writer=HiCacheFile(cfg,file_path=directory)
            writer.register_mem_host_pool_v2(host,PoolName.KV)
            result=writer.batch_set_v2([PoolTransfer(name=PoolName.KV,host_indices=ids,keys=['abc'])])
            self.assertEqual(result[PoolName.KV],[True])
            import subprocess, sys, json
            from dataclasses import asdict
            code = """import json,sys,torch
from sglang.srt.mem_cache.hicache_storage import HiCacheFile,HiCacheStorageConfig
c=HiCacheFile(HiCacheStorageConfig(**json.loads(sys.argv[1])),file_path=sys.argv[2])
x=c.get('abc',torch.empty(int(sys.argv[3]),dtype=torch.uint8))
print(bytes(x.tolist()).hex())
"""
            restored=subprocess.check_output([sys.executable,'-c',code,json.dumps(asdict(cfg)),directory,str(expected.numel())],text=True)
            self.assertEqual(restored.strip(),bytes(expected.tolist()).hex())
            fresh,_=fixture()
            reader=HiCacheFile(cfg,file_path=directory)
            reader.register_mem_host_pool_v2(fresh,PoolName.KV)
            self.assertEqual(reader.batch_exists_v2(['abc']).kv_hit_pages,1)
            result=reader.batch_get_v2([PoolTransfer(name=PoolName.KV,host_indices=torch.arange(12,16),keys=['abc'])])
            self.assertEqual(result[PoolName.KV],[True])
            self.assertTrue(torch.equal(fresh.get_data_page(12),expected))
            del os.environ['SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR']


if __name__=='__main__':unittest.main()
