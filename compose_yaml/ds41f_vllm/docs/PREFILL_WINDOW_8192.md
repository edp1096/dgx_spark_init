# 8192-token prefill window — serving default, 2026-09-13

The default now combines the native Engram reader, an 8192-token prefill window,
4096-token startup tuning, and the qualified dense GEMM profile. Talk packages
the same runtime and profile. For the previous window, use
`DSV41_MAX_BATCHED_TOKENS=4096 ./manage.sh start` after stopping both ranks.

This is a bounded extension of the existing layer loop, not a full port of the
Layered Prefill scheduler. A layer sees up to 8192 tokens before advancing to
the next layer. `SlotLayer.execute` still uses 2048-token expert kernels, keeping
each loaded expert group resident across those kernel calls. Longer inputs
still require multiple windows; there is no per-layer suspension/resumption
across an arbitrary full prompt.

Original weights, the pinned `deepseek_v41` numerical recipe, TP2, DSpark5,
224/128 expert slots, KV2GiB, shared graph I/O and scratch384MiB are unchanged.
The qualified native Engram reader is used in both window-size comparisons.

## Startup memory

The initial 8192 probe was stopped by the external watchdog during FlashInfer
autotuning at 7.68GiB available RAM. There was no host reboot or OOM. The guard
floor remained 8GiB, with a 98GiB cgroup trip threshold and 100GiB hard cap.

`autotune_window.py` limits only FlashInfer's startup dummy tuning requests to
4096 tokens, restoring the upstream hook in a `finally` block. It does not
modify scheduler capacity or actual requests. The original hook and smaller
tuning shapes are preserved. This avoids profiling every large GEMM tactic
while the full expert cache is resident; uncalibrated larger shapes can use
FlashInfer's heuristic. The performance measurements below include that choice.

With `DSV41_AUTOTUNE_TOKENS=4096`, the same guarded 8192 configuration booted
successfully. Head/worker minimum available memory during that startup was
10.19/11.23GiB. No expert slots were removed. Subsequent inference and long
context checks passed. The final profiled qualification had minimum available
RAM 10.57/11.73GiB and peak cgroup use 91.59/91.23GiB, with no OOM or guard trip.
The pre-start estimates retain 116GiB for head and 115GiB for worker, covering
the measured peak deltas plus at least 8GiB. Talk's component estimates are
108/107GiB, separately from the application's configured reserve.

## Initial capped-window throughput

Three timed runs per fixture after an excluded warmup. Same fixed requests,
224 seeded expert slots before each request, no concurrent inference and unique
KV namespaces; all responses are exact and normally terminated. The 4096 reader
A/B used one process, then the 8192 window used a separate boot on the same
6.17 kernel and pinned serving image. PP means input tokens / TTFT, not isolated
GEMM throughput. Startup tuning/preload time is excluded.

| Input | Original Python / 4096 | Native / 4096 | Native / 8192 |
|---|---:|---:|---:|
| 6308-token pp | 298.06 | 314.31 | 422.98 tok/s |
| TTFT | 21.163 | 20.069 | 14.913 s |
| 12934-token pp | 311.91 | 332.56 | 435.42 tok/s |
| TTFT | 41.467 | 38.892 | 29.705 s |

The window alone improves pp 34.57%/30.93%; combined with the reader change,
pp improves 41.91%/39.60%. Timed 8192 TTFT ranges were 14.869–14.929s and
29.692–29.747s. Median expert reads per rank fell 88.05→51.92GiB and
168.31→102.93GiB. The short 595-token input does not benefit from the larger
window; its 577-token copy output had tg45.70, versus 46.08 with native4096 and
45.52 with Python4096. Treat this as maintained decode speed, not a tg gain.

## Context and continuation

- 54,988 input tokens with an 8192-token output allowance returned the exact
  middle marker with HTTP200 and normal stop. Its initial 47.79s latency is a
  capacity check, not a comparative throughput claim. The controlled comparison
  and its resolution are recorded below and in [dense tuning](DENSE_PREFILL_TACTICS.md).
- A 16,211-token record input succeeded. The next turn retrieved its secret
  phrase while reusing 16,128 cached tokens. Changing the secret in record777
  correctly returned the new phrase with 13,824 cached tokens. A subsequent
  short request also succeeded.

Evidence: `results/prefill-window/8192-cap.json`, `comparison.json` and
`uncapped-autotune-guard.json`. Raw telemetry and additional validation records
remain in `~/.local/state/ds41-probes/20260913-window8192{-cap}` and
`20260913-speed`. Tools: `launch_probe.py`, `tools/bench_engram_tp2.py`,
`tools/check_context_capacity.py`, `tools/check_prefill_window.py`.

## Final profiled results

The same three-run test was repeated with all eight dense profile entries
verified on both ranks before every request. All expected responses and paired
output hashes still match. `final-comparison.json` contains the final values.

| Input | Original Python4096 pp → final | TTFT | pp gain |
|---|---:|---:|---:|
| 6308 tokens | 298.06 → 455.35 tok/s | 21.163 → 13.853s | +52.77% |
| 12934 tokens | 311.91 → 461.16 tok/s | 41.467 → 28.047s | +47.85% |

TTFT fell 34.54%/32.36%. Timed ranges were 13.785–13.859s and 28.031–28.067s.
The 577-token copy output had tg45.28 versus the original45.52 (-0.54%),
effectively unchanged in this small test. Four-token answers in the two PP
fixtures have lower displayed tg and are not representative decode benchmarks;
their overall completion latency is still substantially lower.

The controlled repetitive 54,988-token input initially regressed to ~48.44s
with an untuned8192 window. After dense tuning, two timed requests were
38.367/38.499s (mean38.433), versus native4096's38.869/38.739s (mean38.804).
Thus that regression was removed; there is no broad throughput-gain claim for
low-diversity repeated text. Final continuation/invalidation and short-after-long
checks also passed with the profile active.

The production-mode boot disables benchmark RPC/dev mode and includes worker
ASR plus head Extra services. The rebuilt Talk binary returned `spark-final-ok`
through its real chat handler and emitted pp/tg/ttft events; its temporary test
session was deleted. That first real Talk request had 6959 uncached tokens,
pp340.67 and TTFT20.43s. It is a different request/initial state from the paired
fixtures, not an additional causal speed comparison. ASR transcribed the
synthetic English test sentence exactly. See `production-summary.json`.
The same production configuration also passed the 54,988-token capacity test.
Minimum available memory including Talk, worker ASR and head Extra was
11.82/10.76GiB, with no OOM or guard trip. Both monitors exited normally and
the peer recorder reported SSH exit0; the services remain running. Evidence:
`production-context.json` and `production-memory.json`.
