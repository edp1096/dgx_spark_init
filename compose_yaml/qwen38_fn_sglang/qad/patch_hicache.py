"""Wire QAD complete-state payloads into the pinned SGLang HiCache constructor."""
import ast
import os
from pathlib import Path

root = Path(os.environ.get('SGLANG_SOURCE_ROOT', '/sgl-workspace/sglang/python/sglang/srt'))
p = root / 'mem_cache/hybrid_cache/hybrid_pool_assembler.py'
s = p.read_text()
updates = {}
marker = '# QAD complete-state HiCache v2'
if marker not in s:
    start = s.index('def build_hybrid_mamba_stack(')
    end = s.index('\ndef build_hybrid_mamba_swa_stack(', start)
    part = s[start:end]
    changes = [
        ('    params: CacheInitParams,\n', '    params: CacheInitParams,\n    qad_pool=None,\n'),
        ('    kv_host_pool = build_kv_host_pool(\n', '''    # QAD complete-state HiCache v2
    from qad_hicache import CompleteHostPool, prepare, budgets
    qad_plan = prepare(qad_pool, params, server_args, storage_backend) if qad_pool is not None else None
    if qad_plan is not None:
        kv_host_size, mamba_host_size = budgets(
            qad_plan, kv_pool, mamba_pool, server_args.hicache_size,
            params.page_size, params.mtp_draft_device_pools,
        )
    kv_host_pool = build_kv_host_pool(
'''),
        ('    entries = [\n', '''    if qad_plan is not None:
        qsa, ple, ratio = qad_plan
        kv_host_pool = CompleteHostPool(kv_host_pool, qsa, compression=ratio, label="qad-qsa-v2")
        mamba_host_pool = CompleteHostPool(mamba_host_pool, ple, label="qad-ple-v2")
        # Keep these larger payloads separate from pre-patch HiCache files.
        model_name = (model_name or "qad") + "-qad-complete-v2"
    entries = [
''')]
    for old, new in changes:
        if part.count(old) != 1:
            raise RuntimeError(f'Pinned HiCache anchor mismatch: {old!r}')
        part = part.replace(old, new)
    s = s[:start] + part + s[end:]
    old = '        host_pool_group, cache_controller = build_hybrid_mamba_stack(\n            params=params,\n'
    new = old + '            qad_pool=kvcache,\n'
    if s.count(old) != 1:
        raise RuntimeError('Pinned Mamba strategy anchor mismatch')
    s = s.replace(old, new)
    ast.parse(s)
    updates[p] = s

p = root / 'mem_cache/hicache_storage.py'
s = p.read_text()
marker = '# QAD file integrity v2'
if marker not in s:
    old = '\n        storage_key = self._log_key(pool_name, key)\n        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())'
    new = '\n        # QAD file integrity v2\n        from qad_hicache import CompleteHostPool, read_file_page\n        if isinstance(host_pool, CompleteHostPool):\n            return read_file_page(self, pool_name, key, host_pool, page_offset)\n' + old
    if s.count(old) != 1:
        raise RuntimeError('Pinned file reader anchor mismatch')
    s = s.replace(old, new)
    ast.parse(s)
    updates[p] = s

p = root / 'managers/cache_controller.py'
s = p.read_text()
marker = '# QAD generic KV integrity v2'
if marker not in s:
    old = '    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):\n'
    new = old + (
        '        # QAD generic KV integrity v2\n'
        '        from qad_hicache import guarded_page_get\n'
        '        if guarded_page_get(self, operation, hash_values, host_indices):\n'
        '            return\n'
    )
    if s.count(old) != 1:
        raise RuntimeError('Pinned generic KV reader anchor mismatch')
    s = s.replace(old, new)
    ast.parse(s)
    updates[p] = s

p = root / 'managers/scheduler.py'
s = p.read_text()
marker = '# QAD unschedulable request guard v2'
if marker not in s:
    old = '        if not recv_req.return_logprob and recv_req.logprob_start_len != -1:\n'
    new = (
        '        # QAD unschedulable request guard v2\n'
        '        from qad_hicache import admission_error\n'
        '        error_msg = admission_error(\n'
        '            self.server_args, len(req.origin_input_ids),\n'
        '            req.sampling_params.max_new_tokens, self.max_total_num_tokens,\n'
        '        )\n'
        '        if error_msg:\n'
        '            req.set_finish_with_abort(error_msg)\n'
        '            self._add_request_to_queue(req)\n'
        '            return\n\n'
    ) + old
    if s.count(old) != 1:
        raise RuntimeError('Pinned generation admission anchor mismatch')
    s = s.replace(old, new)
    ast.parse(s)
    updates[p] = s

# Check all anchors before changing files; interrupted writes are retryable.
for p, s in updates.items():
    p.write_text(s)
print('Installed QAD complete-state HiCache v2' if updates else 'QAD HiCache v2 already installed')
