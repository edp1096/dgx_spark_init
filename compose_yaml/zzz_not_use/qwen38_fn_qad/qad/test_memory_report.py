import unittest
from types import SimpleNamespace as NS
from memory_report import report, GIB


class Pool:
    def __init__(self, sizes): self.sizes = sizes
    def get_kv_size_bytes(self): return self.sizes


class MemoryReportTests(unittest.TestCase):
    def test_mtp_and_shared_recurrence_are_counted_once(self):
        state = Pool(2*GIB); state.mamba_cache = object()
        def runner(load, kv):
            return NS(model_config=NS(hf_text_config=NS(model_type='qwen4_exp_text'),context_len=1048576),
                      weight_load_mem_usage=load,token_to_kv_pool=Pool((kv*GIB,kv*GIB)),
                      req_to_token_pool=NS(mamba_pool=state),kv_cache_dtype='fp8')
        target, draft = runner(75,6), runner(4,.5)
        scheduler = NS(tp_worker=NS(model_runner=target),draft_worker=NS(_draft_model_runners=lambda:[draft]),max_total_num_tokens=1048576)
        torch = NS(cuda=NS(memory_allocated=lambda:97*GIB,memory_reserved=lambda:99*GIB,
                           max_memory_allocated=lambda:98*GIB,max_memory_reserved=lambda:100*GIB))
        r = report(scheduler,torch)
        self.assertEqual(r['draft_load_delta_gib'],4)
        self.assertEqual(r['kv_and_qsa_gib'],13)
        self.assertEqual(r['mamba_cache_gib'],2)
        self.assertEqual(r['cuda_reserved_gib'],99) # not 99 + KV + load deltas
        scheduler.draft_worker = None
        r = report(scheduler,torch)
        self.assertEqual(r['draft_load_delta_gib'],0)
        self.assertEqual(r['kv_and_qsa_gib'],12)


if __name__ == '__main__': unittest.main()
