# DeepSeek V4.1 Flash on two DGX Sparks

Run the commands below from `compose_yaml/ds41f_vllm/`, the recipe root.

- `docs/`: usage, implementation notes and experiment reports.
- `tools/`: benchmarks, checks, tests and result analysis.
- `patches/`: vLLM overlays.
- `results/`: recorded measurements and logs.
- Root Python modules and launch/build files: serving runtime and deployment.

Method 1 is running: original expert weights and Engram tables are backed by
local SSDs on both nodes. No additional lossy weight quantization is applied.
The current backend uses native **b12x V4.1 MXFP4/MXFP8**, fixed expert slots,
expert CUDA graphs and DSpark with five draft tokens. Expert misses use
O_DIRECT reads into CPU-addressable CUDA storage on GB10. Actual-route reads
are submitted as a native batch and overlap the independent shared expert.

- Head: `192.168.100.61`, worker: `edp1096@192.168.100.60`.
- API: `http://192.168.100.61:8010/v1`, model `deepseek-v4.1-flash`.
- RoCE: `10.200.0.1/24` / `10.200.0.2/24`, `enp1s0f1np1`, `rocep1s0f1`.
- Image: `dgx-ds41-stream:b12x8` on both nodes.
- Context 65,536, one concurrent request, 4,096-token prefill capacity, 2 GiB KV.
- Expert calculations use native 2,048-token kernels with shared graph I/O
  and 384 MiB scratch. Resident experts are consumed before loading missing
  groups to avoid reading them again from SSD.
- Target slots are chosen from available memory, capped at 224 per layer.
  DSpark layers retain up to all 128 experts. Startup refuses an insufficient
  memory budget instead of stopping another model or forcing allocations.

## Run

From this directory on the head:

```sh
./manage.sh status
./manage.sh stop
./manage.sh start
./manage.sh logs
```

`start` reapplies the recipe's dedicated RoCE addresses if missing after a
host reboot, checks both ranks are stopped, then starts worker followed by
head. It does not stop other services. Supported overrides are forwarded to
the worker, for example `DSV41_SPEC_TOKENS=0 ./manage.sh start` disables DSpark.
`DSV41_SLOTS_PER_LAYER`, `DSV41_MAX_BATCHED_TOKENS`, `DSV41_KERNEL_TOKENS`,
`DSV41_SHARED_BUFFERS`, `DSV41_SCRATCH_MIB`, `DSV41_READ_THREADS`,
`DSV41_EXPERT_GRAPHS`, `DSV41_SLOT_IO`, `DSV41_PREFETCH_TEST`, `DSV41_IMAGE` and `DSV41_MOE_BACKEND`
are also forwarded. `DSV41_EXPERT_IO` selects `serial`, `overlap`, `batch` or
the new default `batch_overlap`. `DSV41_CACHE_LAYOUT` optionally selects a
per-layer slot allocation file; the default remains uniform.
`DSV41_SLOT_IO=buffered DSV41_EXPERT_IO=serial` selects the previous transport
for comparison; the default is `direct`.
Code and images must already be synchronized to the worker.
Per-request timing and cached-token details are enabled for SparkTalk's
`pp`, `tg`, `ttft` display (`--enable-per-request-metrics` and
`--enable-prompt-tokens-details`). These are ordinary engine statistics,
not the optional synchronized step profiler.
`DSV41_PREFIX_CACHE_INTERVAL=128` retains periodic sliding-window cache
checkpoints for requests that share only part of their input. Set it to `0`
to reproduce the previous policy. It is forwarded to both ranks and uses
the existing KV allocation. SparkTalk also keeps changing recalled context
after its stable instructions and tools; see
[the implementation and measurements](../../../util/talk/docs/prefix-cache-latency.md).
`DSV41_MAX_MODEL_LEN` and `DSV41_KV_CACHE_BYTES` override context capacity and
KV allocation on both ranks. The 65,536-token serving default matches the
current SparkTalk context and leaves room for its 8,192-token output budget.
The startup memory guard includes KV bytes above the original 512 MiB budget.
To reproduce earlier performance experiments, explicitly set
`DSV41_MAX_MODEL_LEN=2048 DSV41_KV_CACHE_BYTES=536870912 DSV41_PREFIX_CACHE_INTERVAL=0 DSV41_MAX_BATCHED_TOKENS=512 DSV41_KERNEL_TOKENS=512 DSV41_SHARED_BUFFERS=0 DSV41_SCRATCH_MIB=256`.
Historical results also predate resident-first expert grouping; reproducing
their exact code path requires the original source revision. See
[PREFILL_BATCHING.md](PREFILL_BATCHING.md) for the uncached PP comparison,
numerical checks and memory conditions behind scheduler batching. The current
shared-buffer/native-kernel changes and rejected experiments are in
[PREFILL_ARCHITECTURE.md](PREFILL_ARCHITECTURE.md). A monitored 4,096 retry
subsequently passed two boots, paired PP measurements and actual SparkTalk/long
context checks; see [PREFILL_4096_RETRY.md](PREFILL_4096_RETRY.md). The scheduler
default and qualified cap are now 4,096; expert kernels remain at most 2,048.

The API enables automatic tool choice with `--enable-auto-tool-choice`,
`--tool-call-parser deepseek_v41` and `--reasoning-parser deepseek_v41`.
SparkTalk sends `tool_choice: auto` whenever tools are available, including
ordinary questions that ultimately need no tool. Without these launch options,
vLLM rejects such requests with HTTP400 before generation. Thinking remains
off by default; the matching reasoning parser separates it when a request
explicitly enables thinking. See the [vLLM tool-calling documentation](https://docs.vllm.ai/en/latest/features/tool_calling/).
The 2026-09-12 fix passed four live SSE checks: ordinary Korean conversation
with auto tools, automatic function name/ID/JSON arguments, a tool-result
round trip returning42, and reasoning/content separation returning6 with
`tool_choice: none`. Results are in `results/tool-calling-validation.json`;
`tool-calling-runtime.json` records both ranks' applied options. SparkTalk
code did not require changes for this HTTP400 error. A subsequent output-budget
error exposed the old 2,048-token benchmark context limit; normal serving now
uses 65,536 tokens. See [SPARKTALK_INTEGRATION.md](SPARKTALK_INTEGRATION.md).

The second performance round measured five candidates. Batched, overlapped
expert reads improved throughput by **9.3%** on four varied requests and
**6.4%** on four held-out prompts with immediate repeats. Generated outputs,
expert misses and read bytes were identical in the I/O comparisons. Uniform
cache allocation, NCCL, default b12x scheduling and five DSpark draft tokens
remain selected. See [PERFORMANCE_ROUND2.md](PERFORMANCE_ROUND2.md) for each
candidate's measurements, limitations and reproduction commands.

Engram prestaging and model graphs were also implemented and measured, but
remain **disabled by default** because they did not improve these TP2 tests.
The most complete graph variant slowed immediate repeats by about 13%.
See [MODEL_GRAPHS.md](MODEL_GRAPHS.md) for the implementation, measurements,
validation and opt-in flags. The original experiment was packaged as
`b12x7-graphs`; current `b12x8` includes that optional code too. Set
`DSV41_EXPERT_IO=serial` for model-graph or forecast-prefetch experiments.

Predictive expert prefetch is implemented as an opt-in experiment and remains
**disabled by default** (`DSV41_PREFETCH_TEST=0`). Controlled tests found about
2% faster changing requests, 4.7% more expert I/O and 6.5-6.9% slower immediate
repeats with populated caches. See [PREFETCH_EXPERIMENT.md](PREFETCH_EXPERIMENT.md)
for the predictor, correctness tests and measurements. With the flag off,
the early-router hook does no prediction and allocates no spare expert slots.

`launch.sh 0|1` is the single-node primitive. Defaults select the working b12x
configuration. The old Marlin path remains available with
`DSV41_MOE_BACKEND=marlin DSV41_SPEC_TOKENS=0 DSV41_EXPERT_IO=serial`;
it is a diagnostic fallback.

## Weight storage

The Hugging Face revision is `dba1be0a40aa45a94ad051997016db3960a90277` of
`deepseek-ai/DeepSeek-V4.1-Flash`. Each node keeps its full ~510.3 GB snapshot
under `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash`.
Mount the repository root, not just `snapshots/<revision>`: snapshot files
are relative symlinks into `blobs`.

`pack_experts.py MODEL OUTPUT --rank 0|1` prepares the original expert bytes
once into the pinned b12x QMMA layout. Each node's prepared files consume
about 148 GB in `~/.cache/ds41-packed/rank0` or `rank1`. Files are committed
atomically per layer; rerunning resumes completed layers. Headers record the
revision, rank, format, tensor layout and per-expert checksums. Loading rejects
wrong ranks, incompatible layouts and incomplete files.

Actual safetensors payload (all 48 headers inspected):

| Component | Bytes |
|---|---:|
| Routed experts, including MTP | 295,997,276,160 |
| Engram tables and scales | 202,758,032,400 |
| Other weights | 11,530,714,440 |

These are disk sizes, not GPU allocation sizes.

## Runtime changes

- `expert_store.py`: reads TP slices without allocating entire shards. W1/W3
  split rows; W2 splits packed columns, preserving E8M0 scale alignment.
- `pack_experts.py` / `b12x_layout.py`: move repacking out of inference and
  preserve the native V4.1 activation rounding and router-weight placement.
- `b12x_slots.py`: fixed tensors keep their CUDA addresses across cache changes.
  On GB10, b12x's registered and locked shared pool lets O_DIRECT read missing
  expert bytes straight into the final storage. Four readers operate in parallel
  without a file-cache copy or host-to-device staging copy. Each reader owns an
  8 MiB alignment scratch buffer. The native batch reader validates destination
  ownership and fences prior consumers once per batch before overwriting slots.
  Required experts are protected against eviction during each computation.
  The optional buffered path retains its bounded pinned-memory staging window.
- `expert_io.py`: submits descriptors for all actual misses to the pinned b12x
  batch executor, with four native readers. A dedicated CUDA stream waits for
  prior consumers before shared computation is queued on the main stream.
  Reads and shared computation overlap; requested slots become visible only
  after reads finish. The FP32 routed-plus-shared addition order is preserved.
  This does not predict routes or issue additional reads.
- Prefill is grouped by experts when the routing set exceeds cache capacity.
  Each required expert is read once per full prefill batch; inactive routes
  are masked with -1. Recent frequent experts are left resident for decode.
- Expert execution uses bounded token buckets and CUDA graph replay. Padded
  rows have zero input/weights and -1 routes; outputs are trimmed to live rows.
- Scratch is a fixed 384 MiB arena per CUDA stream shared across sequential
  layers. Graph I/O is shared per stream, token capacity and top-k. Borrowed
  outputs are consumed before another layer reuses the buffer. Stream changes
  are ordered by events before slot reuse. Graph arenas never resize while live.
- Direct expert reads bypass Linux's file cache. Buffered expert reads release
  their file-cache pages; checkpoint page cache is released after loading.
  These operations target this model's files;
  they do not globally drop system caches or delete any weights.
- Target and DSpark layers share the same storage mechanism. Native b12x MoE
  accumulates in FP32, includes the shared expert and TP reduction before the
  final BF16 cast.

Engram defaults to the adapted positional-read implementation. The full vLLM
model defaults to eager mode: its "Cudagraph disabled" startup message does
not refer to the separately captured expert graphs. Optional model graphs
passed the limited output and context-transition checks in `MODEL_GRAPHS.md`,
but are not selected for normal serving. Vision and concurrent serving have
not been qualified here. `tools/check_tools.py` checks the SparkTalk SSE request
format with available tools, automatic function selection, a local arithmetic
result round trip and separately streamed reasoning. This limited check does
not qualify every SparkTalk tool or long tool histories.

At 224 target slots, expert tensors occupy about 81.8 GiB per rank including
DSpark. Other weights initially loaded at 6.86 GiB. Scratch, I/O buffers,
CUDA runtime and host processes are additional. Docker's displayed memory
omits significant GPU allocation on this Spark setup; use host memory and
CUDA memory readings when checking headroom.

## Measurements and validation

Results are JSON files in `results/`. `tools/smoke.py` uses the server's token counts
and stream token IDs. Decode rate excludes the first content event's tokens;
end-to-end rate and TTFT are recorded separately. This avoids treating a
speculative multi-token burst as a single token. These are serial smoke tests,
not broad benchmark or model-quality parity claims.

The b12x5 transport comparison restarted both nodes for each mode, retaining
224 slots per target layer and the same DSpark configuration. Each mode then
received the same four distinct prompts in the same order, without immediate
repeats. Persistent kernel compilation caches were retained. The comparison
is one serial pass, not a distribution over repeated cold starts.

| Workload | Buffered decode | Direct decode |
|---|---:|---:|
| Python anagram grouping, 78 output tokens | 4.14 tok/s | 11.22 tok/s |
| English explanation, 107 tokens | 3.94 tok/s | 10.71 tok/s |
| Korean explanation, 148 tokens | 4.94 tok/s | 13.49 tok/s |
| PostgreSQL query, 85 tokens | 5.85 tok/s | 15.92 tok/s |

All four outputs were byte-identical. Total request time fell from 113.82 s
to 42.83 s; decode improved 2.71-2.73x. See `direct-comparison.json` and the
paired `buffered-varied-fresh.json` / `direct-varied-fresh.json` files.
`direct-cache-parity.json` additionally verifies that all 154 logged target
steps on both ranks had identical cache hits, misses, read-byte counts and
expert graph counts through the subsequent diagnostic request. This checks
that the direct run did not benefit from a different expert-cache trajectory.

The isolated reader benchmark on both ranks (`slot-io-rank*.json`) reduced
median replacement time for six experts from about 23.3-23.5 ms to 8.1-8.2 ms,
and 24 experts from about 75.4-76.7 ms to 28.7-29.0 ms. Warm graph execution in
that small test was slightly slower with registered storage (about 0.43-0.44
ms versus 0.38 ms), so this is chiefly an improvement to cache-miss requests.
Native FP8/K32 reference, forced eviction, checksum, bucket-padding and
cross-stream tests passed with direct storage.

Synchronized diagnostics (`*-phase-summary.json`) cover 12 six-token target
verification steps, excluding prefill. Mean traversal time fell from about
786 ms to 313 ms. Expert read time fell from 541/605 ms to 151/159 ms on ranks
0/1. Shared-expert, attention and communication work remains; TP timing also
includes waiting for the other rank. These synchronized timings diagnose
phases and are separate from the uninstrumented throughput figures above.

`direct-prefill-marker.json` verifies the exact marker from a 1074-token
prompt across multiple prefill chunks: first output at 8.26 s, total 8.92 s.
After this longer-input test, host memory usage was about 103 GiB on each
node, with 17-18 GiB available. Earlier in the short-request test it was
98-99 GiB with 22-23 GiB available; prefill leaves additional reusable CUDA
allocator memory. Expert slot count and its 81.8 GiB payload did not increase.
The diagnostic marker `profile.enabled` was removed on both nodes afterward.

The earlier b12x4 immediate-repeat test (`b12x4-long-repeat.json`) measured:

| Workload | Before immediate repeat | Immediate repeat with expert cache populated |
|---|---:|---:|
| Binary-search code, 114 output tokens | 4.26 tok/s | 34.26 tok/s |
| Explanatory prose, 103 output tokens | 3.42 tok/s | 15.96 tok/s |

These are decode rates. They are not cold-start guarantees: cache misses and
DSpark acceptance materially change speed. Prefix-cache hit rate was zero in
these tests. `b12x-prefill-marker.json` records retrieval of an exact marker
from a 1074-token prompt, exercising multiple 512-token prefill chunks.
The marker test reached its first output in 15.85 s. The generated binary
search passed empty/found/missing/single-element cases; repeat outputs were
identical (`final-validation.json`). After these tests, both hosts used about
104 GiB system-wide and retained about 17 GiB available. This includes the OS
and other host processes, not just model tensors.

Validation programs:

- `tools/test_expert_store.py MODEL`: exact TP bytes against safetensors on both ranks;
  cache hit, reload and eviction.
- `tools/test_engram_disk.py MODEL`: exact rows at zero and nonzero rank offsets in
  both Engram tables.
- `tools/test_b12x_stream.py MODEL`: native V4.1 MoE against an independent FP8/K32
  reference, including native rounding and router-weight placement.
- `tools/test_b12x_slots.py MODEL`: graph/eager parity, forced eviction, grouping,
  bucket padding, shared scratch across layers/shapes and stream transitions.
- `tools/test_packed_reader.py MODEL`: multiple staging windows and byte equality
  after full slot eviction, with packed-file checksum verification enabled.
- `tools/benchmark_slot_io.py MODEL`: compares both transports on identical miss
  sequences and checks bit-identical warm outputs. Use at least 24 slots;
  qualification used 32 slots, on both ranks. `DSV41_TEST_RANK=1` selects rank 1.
  Set `DSV41_SLOT_IO=direct` for the slot/reader correctness tests above.
- `tools/check_prefill.py`: long input, exact marker retrieval.
- `tools/test_expert_io.py MODEL`: exact output/checksum/LRU parity, actual read/compute
  overlap, grouping, stream ordering, callback-error drain and read failures.
- `tools/bench_graphs.py`: controlled expert-slot resets, excluded warmup, timed
  I/O modes and telemetry. Requires `DSV41_BENCH_CONTROL=1` at server startup.
- `tools/compare_dspark.py`: token-aware comparison of completed 2/3/5 draft tests.
- `step_profile.py`: opt-in synchronized decode diagnostics when
  `profile.enabled` exists on both nodes, including speculative verification
  batches up to 16 tokens. Keep it absent for throughput timing tests.

## Build and provenance

From the pinned base, build the FlashInfer-enabled image and then b12x:

```sh
docker build -t dgx-ds41-stream:marlin1 .
docker build -f Dockerfile.b12x -t dgx-ds41-stream:b12x8 .
```

`Dockerfile.b12x-runtime` is the incremental recipe when the locally prepared
`dgx-ds41-stream:b12x-dev` image exists. It explicitly preserves NCCL 2.30.7;
a dependency resolver otherwise downgrades it even though b12x does not need
the older library. Torch remains 2.13.0+cu130.

- Base: `vllm/vllm-openai:deepseekv41-flash-0909-arm64`, digest
  `sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58`.
- vLLM: `0.1.dev20904+g179dd0fa9`.
- b12x: https://github.com/local-inference-lab/b12x at
  `789bbb3c846565c41f3404af3e0d7c9ce8702f7f`.
- Engram reader and GB10 attention/indexer fixes derive from
  https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark at
  `458fadec6106e6fd84f7eda36dd6e0a8fa219ef6`. The Engram patch was adapted to
  the pinned image's DP-aware lookup/preparation interface; existing SPDX
  headers are preserved.

Changing the prepared-weight ABI requires new packed files and validation.


The final-layer routed-expert experiment is documented in
[DECODER_ROWS_EXPERIMENT.md](DECODER_ROWS_EXPERIMENT.md). It preserves DSpark and
prefix-cache dependencies and gained about 2% PP, but lost 7.96% matched-output
TG. `DSV41_FINAL_DECODER_ROWS` therefore remains **0 by default**. The trial is
not part of the recommended serving settings.
