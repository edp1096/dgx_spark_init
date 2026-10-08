"""Bound startup-only tuning work without changing the serving token capacity."""
from contextlib import contextmanager
from importlib import import_module
import os
from pathlib import Path


def load_dense_profile():
    if os.environ.get('DSV41_DENSE_PREFILL_PROFILE', '0') == '1':
        from flashinfer.autotuner import AutoTuner
        profile = Path(__file__).with_name('dense-prefill-profile.json')
        # Upstream validates GPU and backend versions and revalidates each
        # tactic at dispatch. Loading replaces only these eight entries;
        # the normal local cache still supplies all other shapes.
        if not AutoTuner.get().load_configs(str(profile)):
            raise RuntimeError('Dense prefill profile does not match this image/GPU')
        print('DS41_DENSE_PREFILL_PROFILE loaded: 8 original MXFP8 GEMM shapes', flush=True)


@contextmanager
def bounded_autotune():
    limit = int(os.environ.get('DSV41_AUTOTUNE_TOKENS', '0'))
    if limit == 0:
        yield
        load_dense_profile()
        return
    if limit != 4096:
        raise ValueError('Only the 4096-token autotune qualification cap is supported')
    module = import_module('vllm.model_executor.warmup.kernel_warmup')
    original = module._flashinfer_autotune_token_counts

    def counts(runner):
        selected = tuple(dict.fromkeys(min(n, limit) for n in original(runner)))
        print(f'DS41_AUTOTUNE_WINDOW tokens={selected}; serving capacity unchanged', flush=True)
        return selected

    # Worker startup is synchronous. Restore the upstream hook even if tuning
    # fails; scheduler config and actual request execution are never modified.
    module._flashinfer_autotune_token_counts = counts
    try:
        yield
    finally:
        module._flashinfer_autotune_token_counts = original
    # Some upstream warmups enter autotune(cache=...), which clears file
    # configs. Apply the qualified large-shape profile after ALL such warmups.
    load_dense_profile()
