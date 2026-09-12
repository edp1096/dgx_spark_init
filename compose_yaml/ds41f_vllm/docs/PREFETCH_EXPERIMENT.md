# Expert prefetch experiment

**Decision: keep predictive prefetch disabled by default.** The implemented
predictor preserves outputs but gives only a small gain on changing requests
and slows cache-resident repeats. This conclusion applies to this predictor
and timing window, not all expert-prefetch designs.

| Measurement | Prefetch off | Prefetch on |
|---|---:|---:|
| Four held-out requests, mean total time over two trials | 43.11 s | 42.24 s |
| Same suites, mean logical expert bytes read on rank 0 | 135.71 GB | 142.10 GB |
| Immediate code repeat, decode | 44.48 tok/s | 41.59 tok/s |
| Immediate prose repeat, decode | 19.89 tok/s | 18.53 tok/s |

Held-out end-to-end speedup was 1.0206x, with approximately 4.71% more reads.
The two cache-resident repeats slowed by 6.50% and 6.86%. All paired outputs
were identical across 24 timed requests; this is a small controlled experiment,
not a broad model-quality or workload-distribution evaluation.

The selected policy's actual useful-prefetch fraction was about 80% on the
held-out requests. However, only 70-85 of 2,674 useful prefetches per trial had
completed when demanded (2.6-3.2%). Most reads still needed a final wait. The
same-layer attention window is short; auxiliary prediction costs remain even
when the cache already contains the useful experts. Starting predictions
earlier or gating them using cache-miss history would require another trial.

The default launch keeps predictive prefetch disabled. This experiment adds
`DSV41_PREFETCH_TEST=1` and uses `prefetch-control.json` to switch modes between
completed requests. Actual routing, attention, expert arithmetic and TP
reduction remain unchanged.

## Predictor

`patches/target_model.py` calls `expert_prefetch.before_attention` in the V4.1
decoder, before attention executes. The earlier V4 decoder is a different
implementation and is not the hook point.

The initial probe recorded two forecasts: the existing gate applied to the
attention input, and a forecast respecting V4.1's delayed mHC residual mixing.
The latter computes the known residual contribution with a zero attention
output, collapses it with the already-available attention pre-mix, applies
the FFN RMSNorm and existing gate, then ranks sqrt-softplus scores plus the
original correction bias. Only this auxiliary prediction assumes zero
attention output; the actual attention and original router still execute.
No predictor training or weight quantization was introduced.

The four `varied` prompts supplied 5,280 six-token layer observations, with
9,074 actual cache misses. Using each predicted token's top six candidates
and a two-expert read budget, residual-aware prediction had 55.88% precision
and recalled 45.43% of misses. The plain attention-input predictor achieved
26.73% precision and 28.08% miss recall. These are forecast measurements,
not measured speedups.

`prefetch-selected-policy.json` fixes the residual predictor, budget two,
and the 21 layers whose probe precision reached 60%. Subsequent speed trials
use different prompts; the policy is not adjusted using those outputs.

## Storage and ordering

Target layers retain 224 resident cache entries and two additional physical
slots. Draft layers keep the previous layout. The extra payload is
752,025,600 bytes per rank, about 0.70 GiB. Expert graphs address the entire
fixed allocation and receive remapped physical slot indices.

Forecasts use O_DIRECT into the spare slots. They do not evict resident
experts. On an actual hit, the completed spare becomes a resident entry and
an ordinary eviction victim's physical slot becomes a spare. An unfinished
useful read is awaited before its weights can execute. Unneeded queued reads
are cancelled; running reads retain exclusive ownership of their spare until
completion. Read errors propagate instead of publishing incomplete weights.

There are two prefetch worker threads and at most four outstanding forecast
futures globally, separate from the four demand-read workers. A dedicated
CUDA stream waits for previous slot consumers without waiting for upcoming
attention. Global CUDA graph capture first quiesces native readers. Cache
reset drains reads before resetting physical-slot ownership.

`tools/test_expert_prefetch.py` checks exact outputs, ready/late promotion, stream
transitions, GPU graph replay while unused spare reads execute, reset with
outstanding reads and injected I/O failure. The result is recorded in
`results/prefetch-correctness.log`.

## Reproduction

Start both nodes with the synchronized experimental source:

```sh
DSV41_MAX_MODEL_LEN=2048 DSV41_KV_CACHE_BYTES=536870912 DSV41_MAX_BATCHED_TOKENS=512 \
DSV41_PREFETCH_TEST=1 DSV41_EXPERT_IO=serial ./manage.sh start
python3 tools/bench_prefetch.py --warmup --suite heldout --name prefetch-heldout
```

The benchmark first warms all four held-out prompts without prefetch, then
runs off/on/on/off. A new control epoch resets expert-cache metadata before
each suite. Resident capacity, physical allocations, DSpark settings and
kernel caches are held constant across the timed modes. Forecast selection
does not use the real router result or future trace labels.

`results/prefetch-heldout-runs.json` records durations, per-request decode
rates and expert I/O counters. The script rejects changed outputs. The
recorded `finish_reason` exposes token-limit truncation. Pending unused reads
may finish just after the final counter snapshot, so read-byte counts should
be read together with `prefetch_pending` rather than treated as disk hardware
counter measurements.

The immediate-repeat check used:

```sh
python3 tools/bench_prefetch.py --suite long --repeat 2 --modes off early --name prefetch-repeated
```

See `results/prefetch-heldout-summary.json`, `prefetch-repeated-runs.json` and
their per-request JSON files. These repeat measurements contain one on/off
pair; the held-out test used the four-run order described above.
The original routing probe is archived as `results/prefetch-probe-1.jsonl.gz`.
