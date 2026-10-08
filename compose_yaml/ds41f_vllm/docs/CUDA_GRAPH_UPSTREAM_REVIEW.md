# CUDA Graph source review — 2026-09-13

> Cleanup 2026-09-13: rejected prototype source, experiment-only launch paths and dedicated replay tools were removed. The descriptions and commands below are historical records, not currently available runtime features. Adopted optimizations, safety guards and measurement evidence are retained.

Follow-up measurements are in [URING_AND_CALLBACK_EXPERIMENTS.md](URING_AND_CALLBACK_EXPERIMENTS.md).
Both candidates were tested; the current serving defaults were retained.

This is a source review, not a new GPU benchmark. Serving settings and the
running model were not changed. Previous measurements remain in MODEL_GRAPHS.md.

## Sources inspected

- Tony: https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/fc725ecf10869c184f4347dd73336536d395753c
  - patch/engram.py, patch/model_state.py, patch/attention.py,
    patch/sparse_attn_indexer.py, launch/dsv41-tp4.sh, patch/mounts.txt.
  - docs/RECIPE.md and docs/EXL3-TP3.md.
- Sero: https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000/tree/45f538a18569420721e00e37353f9a7e1af7e5da
  - adapter/engram_backend.py, adapter/row_store.cpp, adapter/sitecustomize.py,
    boot.py, runtime/flash_mla_sm120.py, B12x parity/replay result JSON files.
- Local streaming_moe.py, streaming_graphs.py, engram_stager.py,
  b12x_slots.py, launch.sh; installed b12x/loader/_direct.c and _batch.c.

## Findings

Tony moves Engram hashing, host synchronization and disk retrieval into
prepare_inputs, ahead of graph replay. Persistent rows let the captured
forward avoid Engram host work. Our optional prestager already ports this
approach to the pinned DP-aware V2 interface. Exact DSpark capture sizes and
disabled adaptive verification are also already represented in our experiment.
His resident experts do not need our per-layer SSD cache replacement breaks.
The EXL3 TP3 a6/a7 comparison reports 1.50x C1 aggregate throughput with graphs,
but is not a measurement of original-weight TP2 expert streaming. Large prefill
remains outside those small decode graphs and its speed is essentially unchanged.

Sero puts Engram retrieval into a native cudaLaunchHostFunc callback between
captured D2H indices and H2D row copies. The callback uses only CPU operations:
row cache lookup, pread and memcpy. GPU dequantization/gather follows the callback.
This differs from our Python eager breaks, but Sero also keeps compute experts
resident. Its 64 GiB cache is a host-wide Engram budget, divided across two tables
and four TP ranks; it is not an expert cache or a per-rank 64 GiB allocation.
The B12x io_uring candidate has component/parity/replay receipts, not a qualified
full-model speedup. The launcher enables SGLang bounded decoder SWA replay;
that option alone is not a portable vLLM implementation or a measured isolated gain.

Our target graph experiment has 41 GPU segments and 40 Python expert breaks.
The break calls b12x_slots.apply, which reads routes through Tensor.tolist(),
manages eviction in Python, submits disk work and dispatches an expert graph.
Installed native direct_into and batch execution call cudaStreamSynchronize.
They cannot be called unchanged from a CUDA host callback: CUDA APIs and waits
depending on later CUDA work are forbidden inside that callback.
See https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EXECUTION.html

The old model-graph launch path also rejects shared execution buffers and
scheduled expert I/O; its guarded probe rejects prefill batches above 512.
Current production uses shared buffers, scheduled overlapping I/O and 8192
prefill. Simply enabling the historical graph option is not a matched comparison
against the present baseline and may discard already-qualified improvements.

## Feasible new experiment, not implemented

The useful new direction is a native expert-cache callback with graph-ordered
dependencies, rather than repeating the old segmented Python implementation:

1. Capture route generation and copy actual route IDs to persistent host storage.
2. Native callback protects every requested expert, resolves slots and loads
   misses using prevalidated CPU aliases of the existing shared allocations.
   No Python/CUDA calls, allocation, graph launches or CUDA-dependent waits
   inside the callback. Existing reader synchronization must be replaced by
   explicit graph/stream ownership, not merely deleted.
3. Publish the mapped IDs through stable buffers and capture b12x compute itself;
   do not call the Python apply/replay wrapper from a host callback.
4. Preserve shared scratch lifetime, shared-expert overlap, TP collectives and
   DSpark target/draft dependencies. Keep the original large-prefill path.
5. On read failure, prevent stale/partial slots from reaching GPU consumers.
   Validate eviction, all-hit replay, changing routes, prefix changes, cancellation
   and target/draft transitions before any throughput claims.

Start with a single decode layer against the current path, both all-hit and
forced-miss cases; only proceed to the model if it improves latency with exact
slot bytes and unchanged outputs. Full-model A/B must retain current cache size,
8192 prefill, DSpark5, native_hint and next-chunk Engram prefetch, and compare
pp/tg/ttft, SSD bytes and peak memory. Preserve startup memory guards.

This could reduce Python dispatch and repeated host synchronization on decode.
It cannot remove SSD transfer time, and a host callback still stalls dependent
GPU work until it completes. Callback scheduling, lost overlap or memory cost
could erase the gain. No defensible improvement percentage is available yet.
Neither repository demonstrates an end-to-end graph with SSD expert replacement
in our original-weight TP2 setup. Broad prefill graph capture is not supported
by this review's evidence.
