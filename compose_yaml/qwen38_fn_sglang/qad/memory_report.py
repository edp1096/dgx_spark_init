"""QAD allocation receipt. Categories are observations, never additive peaks."""
GIB = 1024**3


def report(scheduler, torch):
    target = scheduler.tp_worker.model_runner
    if getattr(target.model_config.hf_text_config, 'model_type', '') not in ('qwen4_exp', 'qwen4_exp_text'):
        return None
    wrapper = scheduler.draft_worker
    drafts = list(wrapper._draft_model_runners()) if wrapper is not None else []
    pools, recurrent = {}, {}
    for runner in [target, *drafts]:
        pool = runner.token_to_kv_pool
        pools[id(pool)] = pool
        state = getattr(runner.req_to_token_pool, 'mamba_pool', None)
        if state is not None:
            recurrent[id(state.mamba_cache)] = state
    def size(pool):
        value = pool.get_kv_size_bytes()
        return sum(value) if isinstance(value, tuple) else value
    # Weight load deltas include loader allocations; graph reservations can
    # overlap allocator totals. Keep them labelled and do not sum them again.
    return {
        'schema': 1,
        'unit': 'GiB',
        'context_tokens': int(target.model_config.context_len),
        'capacity_tokens': int(scheduler.max_total_num_tokens),
        'kv_dtype': str(target.kv_cache_dtype),
        'mtp_layers': len(drafts),
        'target_load_delta_gib': float(target.weight_load_mem_usage),
        'draft_load_delta_gib': sum(float(r.weight_load_mem_usage) for r in drafts),
        'kv_and_qsa_gib': sum(size(p) for p in pools.values()) / GIB,
        'mamba_cache_gib': sum(size(p) for p in recurrent.values()) / GIB,
        'cuda_allocated_gib': torch.cuda.memory_allocated() / GIB,
        'cuda_reserved_gib': torch.cuda.memory_reserved() / GIB,
        'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated() / GIB,
        'cuda_peak_reserved_gib': torch.cuda.max_memory_reserved() / GIB,
    }
