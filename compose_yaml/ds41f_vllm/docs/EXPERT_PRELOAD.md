# Frequency-based expert preload

The candidate seeds existing target-expert slots from a checkpoint/layout-bound
routing-frequency profile. It never changes expert weights, routing, the
224-slot target allocation, draft residency, KV capacity or decoder computation.

`DSV41_PRELOAD_COUNT=0` disables startup seeding. `128` or `224` seeds up to that
many experts per target layer. `patches/gpu_worker.py` runs the seed after model
warmup/graph preparation and before worker initialization returns to the engine,
so startup I/O is not counted as request TTFT. It fences CUDA consumers, clears
only target mappings and loads the selected rank's original packed records.
Most frequent experts are placed at the recent end of the initial LRU.

The profile contains IDs only, with checkpoint revision, packed-layout ABI and
training trace SHA256. It is learned from previous short requests, separate from
the A/B evaluation fixtures. Missing layers, duplicate/out-of-range IDs, wrong
checkpoint/layout and a preload count beyond cache capacity are rejected.

During dedicated tests, `DSV41_BENCH_CONTROL=1` enables a worker RPC to seed/reset
caches between trials. That RPC checks the flag before changing anything. The
HTTP development routes are enabled only for the same benchmark flag; normal
serving keeps them disabled. Benchmark API listens on loopback.

`tools/bench_expert_preload.py` measures 0/128/224, alternating order, distinct
KV namespaces for independent trials, and an immediate repeated request to
check prefix-cache reuse. Warmup trials are marked separately. Every response
must match the expected output and finish normally. Request counters detect
another caller during a measurement. Rank-local cache deltas, loading time,
TTFT, TG, token counts and prefix hits are retained.

Before timing, two representative seeded experts in every target layer on each
rank are read back and checked against packed-file checksums (160 slot checks
across TP2). Timing does not include those checksum reads.

Profile regeneration:

```sh
python3 tools/build_expert_hot_profile.py \
  --trace results/io-schedule-varied-warmup-routes-rank0.jsonl \
  --packed-header "$HOME/.cache/ds41-packed/rank0/layer-00.bin" \
  --output expert-hot-profile.json
```

This document records the candidate mechanism. The final decision and measured
results are appended after the controlled trials finish.

## Decision and measured results (2026-09-12)

Enabled **224** by default in the standalone launcher and Talk's embedded DS41
recipe. This changes which expert records fill the existing slots before API
readiness. It does not reserve additional slots. Setting `DSV41_PRELOAD_COUNT=0`
disables it; Talk's DS41 component/binding `runtime_options` accepts the same key
with an integer 0..224. The number of cache slots remains 224 regardless of
preload count. Regenerate/rebuild the embedded recipe when changing its source.

Two interleaved trials per 0/128/224 mode, three input fixtures and first/repeated
requests produced 36 timed responses. Same-process kernel warmups are excluded.
The initial harness incorrectly required a prefix hit on a 157-token request;
that assumption was corrected (short requests can have no reusable KV block),
and all timed comparisons were restarted. No model failure occurred. The
preliminary records are retained separately from the final A/B.

| Input | No preload TTFT | 224 preload TTFT | Reduction |
|---|---:|---:|---:|
| 157 tokens, exact code copy | 9.77 s | 2.90 s | 70.3% |
| 3,522 tokens, exact code copy | 17.52 s | 7.94 s | 54.7% |
| 12,934 tokens, mixed document marker | 53.44 s | 41.91 s | 21.6% |

Median upfront loading cost was **12.63 s** for 224, versus 7.31 s for 128.
Those seconds occur before requests in production, and are **not included** in
the TTFT table. Including preload in the cold-start total, this is principally
a shift of I/O into startup rather than elimination of I/O. The benchmark
baseline explicitly clears expert mappings; it is not a measurement of an
already warmed running session. All modes use the same 87,836,590,080 cache
bytes/rank (~81.80 GiB).

TG is workload-dependent. On the 114-token code output, 224 changed first-call
TG from 26.66 to 29.60 t/s for the short prompt, but from 29.07 to 27.51 t/s
(-5.37%) after the longer prompt. A separate, longer exact-copy check generated
577 tokens per call in two alternating trials: **44.08 -> 43.67 t/s (-0.92%)**,
with TTFT **16.80 -> 8.00 s**. This is not a general decode-throughput upgrade.

After repeating the long mixed-document input, TTFT was about 1.33 -> 1.28 s.
The very short prompt repeated at about 0.67 s in both modes even without KV
hits, demonstrating the effect of the naturally warmed expert cache. Long-copy
repeat TTFT had a baseline outlier (3.0 vs 0.52 s), so its two-sample median is
not presented as a reliable improvement estimate.

All 40 timed responses matched their expected output and stopped normally;
160 representative loaded-slot checksum checks passed across both ranks.
These verify loading integrity and the tested responses, not broad model quality
or exact bitwise equality of every intermediate operation. No expert pruning,
extra weight quantization, cache partition, decoder replay, or prediction-based
prefetch was enabled.

Artifacts:
- `results/expert-preload-checksums.json`
- `results/expert-preload-ab.json`, `results/expert-preload-summary.json`
- `results/expert-preload-long-output.json`, `results/expert-preload-long-output-summary.json`
- `results/expert-preload-warmup-and-short-cache-check.json` (excluded preliminary run)
- `results/expert-preload-final-runtime.json` (production startup and Talk check)

The final production boot was executed through Talk's embedded recipe, not the
workspace launcher. Both ranks logged exactly one 224-expert preload (12.64 /
12.68 s) before readiness. The API's development RPC returned 404 with benchmark
control OFF. A fresh Talk conversation returned the correct answer `15` and
persisted performance metrics (4,361 input tokens, TTFT 19.81 s). This is an
integration check, not another matched A/B row.

External observers recorded no reboot, OOM, guard trip or swap growth. Across
the full A/B, minimum available RAM was 14.58 GiB on the head and 13.14 GiB on
the worker; the final production boot/check stayed above 15.02 / 14.16 GiB.
See `results/expert-preload-memory.json`. Python profile validation and Go
orchestrator/config/server tests passed. The DS41 model set and ASR were stopped
after validation, restoring the user's stopped state; Extra services remain up.
