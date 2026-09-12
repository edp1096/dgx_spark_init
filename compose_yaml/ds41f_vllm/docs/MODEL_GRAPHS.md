# TP2 Engram staging and model CUDA graphs

**Decision: keep all these additions disabled by default.** They preserve the
tested outputs but do not improve the measured TP2 workload. Normal serving
retains the b12x6 eager model with its existing expert graphs and direct I/O.

This ports the Engram preparation and graph approach from
[the DGX Spark TP4 recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
at `458fadec6106e6fd84f7eda36dd6e0a8fa219ef6` to this recipe's pinned
vLLM `0.1.dev20904+g179dd0fa9` and SSD-backed TP2 experts.
The upstream TP4 throughput is not a TP2 performance target: TP4 keeps all
routed experts resident, while this TP2 configuration reads cache misses
from local SSDs.

## Implementation

`engram_stager.py` runs from the V2 model state's `prepare_inputs`, before
the target model executes. It computes the original n-gram hashes from the
actual input tokens and lookback window, copies both tables' rank-local hash
columns to pinned host memory once, batches the original disk lookups, and
copies the resulting BF16 rows into persistent GPU buffers. Hashing, table
selection and dequantization preserve the original operations. It requires
DP1, PP1, no microbatch overlap and the V2 stateless lookback path.

`streaming_moe.py` and `streaming_graphs.py` use segmented CUDA graphs.
GPU operations before and after the routed experts are captured. A replayed
eager function between those segments reads the actual router output,
updates SSD-backed expert slots and executes the existing b12x expert graphs.
Its output buffer is allocated by the caller within the graph memory pool.
Capture only writes a finite placeholder; it must not read router IDs from
prefix kernels that have not executed yet. Normal execution and replay
always use the actual selected experts.

The first PIECEWISE implementation left the sparse attention operation eager
and produced no speed gain. The full-attention implementation uses vLLM's
FULL dispatch and persistent attention metadata, but replaces its monolithic
capture with `BreakableCUDAGraphCapture`. Ordinary attention breaks remain
suppressed in FULL mode; expert I/O uses a mandatory break that also applies
in FULL mode. Six target shapes (1 through 6 tokens) contain 41 GPU segments
and 40 CPU expert breaks each. The five-token DSpark draft graph contains
four GPU segments and three expert breaks. Larger prefills execute eagerly.

Full capture normally disables the indexer's short-context shortcut, adding
work that eager execution avoids. A second set of target graphs preserves
that shortcut for a CPU sequence-length upper bound of at most 512. Each
indexer checks that this bound divided by its compression ratio fits in its
512 selected positions. Every compressed candidate is then selected, using
the same fill kernel as the original eager shortcut; the K cache is still
built. Longer contexts use the general graph. The bound includes scheduled
tokens and may overestimate; underestimating would be incorrect. This is a
runtime context test, independent of the user's prompt content.

The model and worker patches start from this image's sources. In particular,
V4.1 has a separate NVIDIA decoder and a DP-aware Engram `lookup` interface;
copying older upstream whole-file patches would overwrite incompatible APIs.

## Controlled validation

- `tools/test_engram_stager.py`: original lookup versus prestaging, both TP row
  ranges, both tables, changing token counts and lookback windows; exact
  rows and hashes, stable GPU pointers, dummy-mask handling.
- `tools/test_breakable_experts.py`: real expert eviction between captured GPU
  segments, exact outputs, caller-allocated graph output, multiple token
  shapes sharing one graph pool and alternating replays.
- `tools/bench_graphs.py`: warmup excluded, expert caches reset before each timed
  suite, unchanged slot capacity and DSpark settings, GPU telemetry on both
  nodes. `--bypass-graphs` retains staged Engram and graph allocations while
  dispatching target and draft eagerly, isolating graph execution in one process.

Benchmark control requires `DSV41_BENCH_CONTROL=1` and uses
`graph-control.json` on both nodes. It is disabled for normal serving. The
bypass cannot skip startup capture: it applies only after all graphs have
been captured. Remove `profile.enabled` for these tests; synchronized eager
profiling is not compatible with capture or throughput measurements.

The upstream GPU-state probe observed roughly 225/239 GB/s GEMV on the two
nodes and 230/244 GB/s under continuous load. The reported ~70 GB/s slow
state was not observed during this probe. No clocks, power settings, kernel
drivers or other services were changed.

## Results

Each varied suite contains Python, English explanation, Korean explanation
and SQL requests in that order, with 418 output tokens total. The expert
cache starts empty for each suite; compilation and Engram file caches are
warmed. Slot capacity remains 224 per target layer and DSpark remains five
tokens. Baseline and each graph variant have two timed suites after an
excluded warmup. Prestaging alone has one timed suite.

| Configuration | Mean total request time |
|---|---:|
| Original eager model | 42.384 s |
| Engram prestaging, eager execution | 42.036 s |
| Prestaging and PIECEWISE model graphs | 42.937 s |
| Prestaging and full-attention segmented graphs | 42.885 s |
| Full-attention graphs plus short-context variants | 42.836 s |

Prestaging alone differs by only 0.8%, insufficient evidence of a useful
gain in this small test. The complete graph variant is 1.1% slower overall.
All timed varied outputs, token counts and finish reasons match the original
eager reference. All configurations finish with 13,819 expert misses and
129,903,022,080 expert read bytes on rank 0. The available final 93 target
steps of the baseline and first variants have identical cache-counter
trajectories, not just matching final totals.

The final graph process was also tested with eager dispatch versus graph
dispatch while retaining the same allocations and Engram prestaging. Each
mode received an excluded warmup and one timed suite: binary-search code
twice, then explanatory prose twice. The second request for each prompt has
a populated expert cache:

| Immediate repeat | Graphs bypassed | Complete graphs | Change |
|---|---:|---:|---:|
| Binary-search code, 114 output tokens | 43.57 tok/s | 37.87 tok/s | -13.1% |
| English prose, 103 output tokens | 19.42 tok/s | 16.94 tok/s | -12.8% |

Outputs and token counts are identical. Both suites have 10,574 expert misses
and 99,398,983,680 expert read bytes. These are decode rates, with the first
stream content event excluded; they are not cold-request rates.

The context transition test generates from a 500-token prompt to 614 total
tokens. Eager and graph outputs are identical. The 1,074-token multi-chunk
prefill test returns the exact marker, taking 8.72 s to first output and
9.28 s total with graphs enabled. A live Python stack confirms execution
inside the graph replay, including the CPU expert reader break.

Capture memory reported by vLLM is about 0.75 GiB for PIECEWISE, 0.99 GiB for
the head's general full-attention graphs, and 1.57/2.26 GiB on the two nodes
with short-context variants. Stream-local scratch also grows from 256 MiB
in the baseline to 1 GiB across the capture/runtime streams in the final
experiment. These measurements overlap and must not be added as independent
costs. Both hosts retained about 16 GiB available after the complete variant.
The graph startup budget reserves an additional 3 GiB.

During active GPU samples the head reported roughly 2.17-2.18 GHz and the
worker 2.19 GHz, without active clock-throttling flags. Together with the
bandwidth probe, this provides no evidence for the upstream low-clock or
slow-bandwidth condition in this test window. The TP4 guide's resident-expert
speedup does not reproduce with these CPU expert breaks and SSD-backed TP2
slots; graph creation alone is not evidence of a speed improvement.

Raw results include `model-graphs-comparison.json`, `graphs-hot-comparison.json`,
`piecewise-cache-parity.json`, `graph-context-transition.json`,
`graphs-prefill-marker.json`, `graph-gpu-telemetry-summary.json`, per-trial
requests and engine logs in `results/`.

## Reproduce the optional experiment

The normal defaults are unchanged. To explicitly run the complete graph
variant after stopping this recipe's two containers:

```sh
DSV41_IMAGE=dgx-ds41-stream:b12x7-graphs \
DSV41_MAX_MODEL_LEN=2048 DSV41_KV_CACHE_BYTES=536870912 DSV41_MAX_BATCHED_TOKENS=512 \
DSV41_MODEL_GRAPHS=1 DSV41_ATTENTION_GRAPHS=1 DSV41_EXPERT_IO=serial \
DSV41_SHORT_CONTEXT_GRAPHS=1 DSV41_ENGRAM_PRESTAGE=1 \
./manage.sh start
```

`DSV41_ATTENTION_GRAPHS=0` selects PIECEWISE. With attention graphs enabled,
`DSV41_SHORT_CONTEXT_GRAPHS=0` selects only the general graphs. Setting just
`DSV41_ENGRAM_PRESTAGE=1` selects prestaging around eager execution.
All flags are forwarded to the worker. The experiment requires predictive
expert prefetch to remain off. Context 2048, one request and five draft tokens
are the qualified settings.

Controlled benchmarks additionally require `DSV41_BENCH_CONTROL=1` at startup.
Use `tools/bench_graphs.py --label NAME --suite varied --trials 2`, adding
`--bypass-graphs` for eager dispatch. `tools/check_graph_transition.py` also requires
benchmark control. Omit this flag for normal serving; control files then have
no effect. `./manage.sh stop` followed by plain `./manage.sh start` restores
the default execution configuration.
