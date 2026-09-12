# TP2 performance measurements, second round

These measurements used context 2,048 and 512 MiB KV. Subsequent SparkTalk
integration increased normal serving to context 65,536 and 2 GiB KV; see
SPARKTALK_INTEGRATION.md. The speedup figures below describe the original
controlled workloads and are not measurements of 64K-input performance.

The selected improvement is native batched expert reads overlapped with the
independent shared expert. Original MXFP4/MXFP8 weights, routing, uniform 224
expert slots, NCCL, default b12x scheduling and DSpark5 remain selected.
Full-model graphs, Engram prestaging and forecast prefetch remain disabled.

## Fixed conditions and interpretation

Two GB10 DGX Sparks, TP2, context 2048, maxseq 1, prefill 512, 512 MiB KV.
Pinned model revision and dependencies are in README.md. The initial image
was b12x6; I/O trials used b12x8-dev, cache and draft-length trials b12x8-cache.
The recipe is bind-mounted, so final source hashes are recorded separately.
Both nodes retained the original checkpoint and prepared expert bytes.
Target cache budget was exactly 8,960 slots; target plus draft payload was
87,836,590,080 bytes (81.8 GiB) per rank in all server trials.

Each throughput suite has one excluded warmup. Expert slots are reset before
each suite; kernel compilation caches remain. Short prompts do not reuse a
full 128-token prefix block. Four held-out prompts each run twice in immediate
succession: merge-interval code, DNS explanation, Korean cache/database
explanation and monthly SQL. The second request therefore measures a populated
expert cache. Diagnostic synchronization and route tracing are off during
all timed requests. GPU telemetry and both engine logs accompany the JSON.

Speedup is baseline time divided by candidate time. For identical output
counts this is a throughput gain; percentage time saved is a different number.
These are small, serial workload measurements, not universal speed or quality
guarantees. Draft-length trials can change generated wording and token counts;
they require token-aware interpretation.

## 1. Actual-route I/O: adopted

`expert_io.py` uses the existing pinned b12x native batch executor with four
readers. A single Python submission carries descriptors for all actual expert
misses. The native path validates destination ownership, orders reads by file
offset and fences its CUDA stream once for the batch. No new native-library
code, route prediction or extra expert reads are involved.

Before shared computation is queued, a dedicated I/O stream waits for prior
slot consumers. Missing expert bytes are read directly into registered CUDA
slots while shared computation runs. The model waits for all reads before
using the slots. Required experts stay protected, LRU ordering stays unchanged,
and the final FP32 addition remains routed plus shared before TP reduction
and the BF16 cast. Hits retain the previous execution order.

`tools/test_expert_io.py` passed on both ranks: exact outputs/checksums/LRU order,
1/6/17-token shapes, forced eviction and grouped execution, cross-stream use,
callback failure draining, and injected read-error propagation. A gated reader
and bounded GPU delay prove that reads complete while later shared-stream work
is still pending. Reset-and-reuse after the injected callback failure is exact.

Varied suite: timed order serial/overlap/batch/combined/combined/batch/overlap/
serial, two suites per mode, four requests per suite. All 32 timed responses
are byte-identical with matching token counts and finish reasons. Every suite
has 13,819 misses and 129,903,022,080 logical expert-read bytes in the
last head-rank sample. Per-rank final counters match across all arms; the
worker log can include a later discarded speculative step. See
`results/round2-io-parity-audit.json` for both ranks.

| Mode | Mean four-request time | Throughput versus serial |
|---|---:|---:|
| Serial | 42.7569 s | baseline |
| Shared overlap | 41.5699 s | +2.86% |
| Native batch | 40.5200 s | +5.52% |
| Batch + overlap | 39.1064 s | **+9.33%** |

Held-out suite with immediate repeats: serial/combined/combined/serial, eight
requests per suite. All 32 responses and read counters match. Mean time falls
57.3648 to 53.9102 s: **+6.41% throughput, 6.02% less elapsed time**. Final
misses are 14,713 and logical read bytes 138,306,908,160 for every suite.

| Held-out workload | First decode, serial -> combined | Immediate repeat, serial -> combined |
|---|---:|---:|
| Merge intervals | 12.30 -> 13.25 tok/s | 49.33 -> 49.27 tok/s |
| DNS | 8.56 -> 9.44 tok/s | 19.37 -> 19.72 tok/s |
| Korean explanation | 14.31 -> 15.76 tok/s | 29.15 -> 29.12 tok/s |
| SQL | 14.20 -> 15.56 tok/s | 40.65 -> 40.88 tok/s |

Evidence: `results/io-schedule-comparison.json`, `io-heldout-comparison.json`,
all corresponding per-request JSON/engine logs, and `expert-io-test*.log`.
The benefit is concentrated in requests requiring SSD reads; fully resident
repeats are preserved within the observed variation.

## 2. Cache allocation/replacement: keep uniform LRU

`tools/analyze_cache_layout.py` exactly simulates protected routing sets, grouped
prefill and LRU. A bounded knapsack allocates the same 8,960 target slots among
40 layers, trained only on the varied-suite warmup trace. The selected
`cache-layout-trained.json` ranges from 144 to 320 slots per layer. Training
misses fall 13,740 to 13,186 (4.03%). Simulated SLRU protection fractions
0.5/0.75/0.9 reduce misses by 0/0.05/0.64%; SLRU was not added to serving.

Actual held-out allocation trials retain identical outputs and the same exact
cache payload. Misses fall 14,713 to 14,546 (1.14%), but mean time rises
53.9102 to 54.1790 s (0.50% slower). Immediate code repeat falls 49.27 to
46.16 tok/s (6.33%). Keep uniform 224 and LRU. The allocation is an opt-in
experiment through `DSV41_CACHE_LAYOUT`, not the selected serving policy.

Evidence: `results/cache-layout-comparison.json`, `cache-layout-heldout-*`.
The allocation also passed exact marker retrieval from a 1074-token prompt
(`cache-layout-prefill.json`); that single timing is not a controlled speedup.

## 3. TP communication: keep NCCL

`tools/benchmark_tp_collectives.py` compares the pinned vLLM PyNcclCommunicator
(NCCL 2.30.7) with the pinned b12x RoCEnante API1 on the actual dedicated TP2
rail. FP32/BF16 outputs are exact. Four interleaved blocks give 100 timing
samples per mode; reported medians use the slower rank.

| Operation | PyNccl | RoCEnante |
|---|---:|---:|
| FP32 all-reduce, 20 KiB | 29.552 us | 47.936 us |
| FP32 all-reduce, 120 KiB (six target tokens) | 51.568 us | 69.888 us |
| FP32 all-reduce, 256 KiB | 111.152 us | 77.232 us |
| All-gather, about 1.55 MB | 223.152 us | 196.960 us |

The dominant small decode reduction regresses about 35.5%. Larger transfers
can improve, but this does not justify replacing the decode path. No full-model
RoCEnante adapter or end-to-end speedup is claimed. Both temporary processes
exited zero and all output checks passed. Upstream printed an ignored Python
interpreter-teardown destructor exception after explicit close; it did not
occur in the timed operations.

Evidence: `results/tp-collectives-comparison.json` and per-rank logs.
Reproduce with `tools/run_comm_benchmark.sh` while this recipe's server is stopped.

## 4. b12x scheduling: keep default

The pinned V4.1 numerical policy selects dynamic/internal/M64/grouped execution.
This trial changes its supported scheduling limit, not that numerical recipe.
`tools/benchmark_moe_clusters.py` sweeps dynamic max-active-clusters at 8/16/24/32/48,
clearing the planner cache per candidate, capturing each and interleaving
forward/reverse timings. Actual registered checkpoint expert storage is used
on both ranks. Shapes are target1, target6, draft5 and prefill 512. Outputs are
exact for every candidate on both nodes.

Head medians: target1 is 0.40138 ms at 48 versus 0.39789 at 32 (under 1%);
target6 is 1.86016 at 48 versus 1.85853 at 24 (under 0.1%). Draft5 and prefill 512
favor 48 (0.86178 and 12.35322 ms). The worker shows the same lack of a useful
consistent gain. Keep automatic/default 48; no full-server scheduling gain is
claimed and no override is enabled.

Evidence: `results/moe-clusters-rank0.json`, rank1 and logs. Reproduce with
`tools/run_kernel_benchmark.sh` while the server is stopped. This is a supported
scheduling sweep, not an exhaustive search over new kernel algorithms.

## 5. DSpark proposal length: retain 5, optional 3 for prose

Uniform 224 and combined I/O are fixed. Length 5 uses the two combined held-out
trials; lengths 3 and 2 each restart both ranks, then run one excluded warmup
and two timed suites. Runs are ordered 5, 3, 2, not interleaved. All timed replies
finish normally, and each length produces stable text across its two trials.

| Draft length | Mean eight-request time | Output tokens | Aggregate end-to-end rate |
|---|---:|---:|---:|
| 5 | 53.9102 s | 816 | 15.1363 tok/s |
| 3 | 50.3496 s | 796 | 15.8095 tok/s |
| 2 | 54.3648 s | 796 | 14.6418 tok/s |

Length 3's aggregate token rate is 4.45% above 5 on this mix, but it is not a
same-output comparison: DNS changes 101 to 96 tokens, Korean 93 to 88, and SQL
changes function-name case. Merge-interval code remains identical at 142 tokens.
Length 2 and 3 outputs are identical on the tested prompts. All three generated
merge functions pass six empty/overlapping/touching/nested/negative cases;
SQL is identical after case normalization. This is limited functional checking,
not model-wide quality parity. No extra lossy weight conversion is involved;
the cause of the draft-length-dependent wording was not isolated.

| Workload | First decode:5 /3 /2 | Immediate repeat:5 /3 /2 |
|---|---:|---:|
| Merge intervals | 13.25 /12.05 /11.21 | 49.27 /40.26 /35.35 |
| DNS | 9.44 /10.93 /10.36 | 19.72 /25.87 /24.89 |
| Korean explanation | 15.76 /18.09 /15.91 | 29.12 /34.19 /29.01 |
| SQL | 15.56 /15.71 /13.81 | 40.88 /38.89 /33.08 |

Units are output tok/s. Length 3 improves these prose requests, including
+31.18% on repeated DNS and +17.41% on repeated Korean, but identical-code
repeat slows 18.29%. Length 2 offers no winning overall tradeoff here. Preserve
length 5 as the default to retain existing behavior and code performance;
`DSV41_SPEC_TOKENS=3 ./manage.sh start` is an available measured prose option.
No automatic content classifier or per-request draft policy was added.

The pinned adaptive verifier needs full-graph timing samples to initialize its
draft cost curve. It is not qualified on this eager model; launch rejects
`DSV41_ADAPTIVE_VERIFY=1` without model graphs. This is a source-level constraint,
not a measured adaptive result. Prior full-model graph tests regressed; an
adaptive eager implementation would be a separate change.

Evidence: `results/dspark-length-comparison.json`, `dspark-output-checks.json`,
`dspark2-heldout-*`, `dspark3-heldout-*` and the length 5 I/O held-out trials.
Regenerate the token-aware comparison with `python3 tools/compare_dspark.py`.

## Reproduction and selected operation

From this directory on the head, stop only this recipe before changing startup
settings. Wait for stop to finish before start. Worker source and image must
already match. Normal operation uses plain `./manage.sh start` with image
`dgx-ds41-stream:b12x8`, direct `batch_overlap`, uniform 224 when memory permits,
DSpark5, eager full model and enabled individual expert graphs. Benchmark
control defaults off; control files then have no effect.

For the I/O comparison:

```sh
DSV41_MAX_MODEL_LEN=2048 DSV41_KV_CACHE_BYTES=536870912 DSV41_MAX_BATCHED_TOKENS=512 \
DSV41_BENCH_CONTROL=1 ./manage.sh start
python3 tools/bench_graphs.py --label io-recheck --suite varied \
  --io-modes serial,overlap,batch,batch_overlap,batch_overlap,batch,overlap,serial
```

For another held-out comparison after the API is ready:

```sh
python3 tools/bench_graphs.py --label heldout-recheck --suite heldout --repeat 2 \
  --io-modes serial,batch_overlap,batch_overlap,serial
```

For cache allocation, restart with `DSV41_CACHE_LAYOUT=cache-layout-trained.json`
and benchmark control enabled; for draft length, restart with
`DSV41_SPEC_TOKENS=2` or 3, allocation unset. Use the same held-out suite and two
timed combined-I/O trials. Do not compare output time without checking token
counts and text. Stop the benchmark service, archive its control files, and
start without `DSV41_BENCH_CONTROL` to return to ordinary serving.

The incremental final image is built on both nodes with:

```sh
docker build -f Dockerfile.b12x-runtime -t dgx-ds41-stream:b12x8 .
```

No Talk models, unrelated worker jobs, GPU power limits, drivers or checkpoint
contents were changed. This round does not establish concurrent serving,
vision/tool quality, arbitrary contexts above 2048 or a universal optimum.

## Final qualification

Completed 2026-09-12 02:55 UTC (11:55 KST). Both ranks run the final b12x8
image with the selected settings and benchmark control disabled. API health
is 200. Three short responses match the expected Korean capital, addition
function and 1-to-10 list. The 1074-token input returns the exact marker
`spark-cobalt-731` (TTFT 9.799 s, total 10.292 s). This single post-restart marker
run is a correctness check, not a controlled prefill performance comparison.

After these checks, host-available memory is 18.45 GiB on the head and 18.02 GiB
on the worker; expert payload remains 81.8 GiB per rank. Both engine logs show
actual batch/overlap activity, no benchmark resets and no Python traceback.
`results/b12x8-final-validation.json` records the settings checks, memory and
counters; the short/prefill JSON and both final engine logs retain the evidence.
`b12x8-provenance.json` records image identities and matching SHA256 hashes for
57 recipe source files across both nodes. Verified packages are Torch
2.13.0+cu130, vLLM 0.1.dev20904+g179dd0fa9, b12x 1.3.0 and NCCL 2.30.7.

The initial final-validation attempt was interrupted by a head-host reboot
at approximately 2026-09-11 21:07 UTC, before any smoke response. It was not a
successful final check. The remaining worker rank was stopped, the dedicated
rail address was restored by manage.sh, both ranks were started again, and
all final checks above were completed afterward. The interrupted state is
preserved in `results/b12x8-interrupted-validation.json`.
