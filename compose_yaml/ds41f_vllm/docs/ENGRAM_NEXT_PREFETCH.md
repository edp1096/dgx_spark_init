# Next-chunk Engram prefetch

Qualified default on: `DSV41_ENGRAM_NEXT_PREFETCH=1` on both ranks.
Set it to `0` before launching to disable it. Standalone and Talk launchers
use the same packaged implementation.

Forecast only the next bounded chunk of the longest remaining text prompt.
The original V2 stateless n-gram GPU hash kernel receives the actual known
prompt tokens, positions and preceding lookback tokens. Copy only this rank's
hash columns to a separate host buffer and complete that copy before submission.
Before model execution, a single background job de-duplicates owned
rows and reads their original weight and scale bytes with two native workers.
These disposable buffered reads warm the OS page cache; the original demand
read and dequantization remain authoritative. No expert prediction, weight
conversion, KV layout changes, or retained decoded-row cache.

At most one job is outstanding. A busy job prevents another forecast; forecast
length is bounded by scheduler capacity (8192). Prompt completion, decode,
excluded multimodal/prompt-logprob requests do not start forecasts. A cancelled
request can leave one disposable read in flight, never a growing queue.

The benchmark-only control switches the same process between on/off after
joining any outstanding job. It measures demand-stage and forecast work and
checks forecast hashes against the actual next chunk in a separate validation
mode. Timed runs disable that extra hash-comparison synchronization. A failed background
read disables prefetch without changing demand reads. Benchmark RPCs require
`DSV41_BENCH_CONTROL=1`; production keeps that disabled.

## Validation

Measured on 2026-09-13, original TP2 checkpoint/b12x, 224 target slots,
128 draft slots, DSpark 5, 2 GiB KV/rank, 8192 scheduler, 2048 expert kernel,
original 8 dense tactics. The previous C native demand reader stays selected.
One process, alternating on/off order, identical requests, expert-cache seed
before each request, distinct KV cache salt, zero reused prompt KV tokens.
Each fixture has one excluded warmup per variant and three measured trials.
Hash-verification synchronization is disabled in timed runs.

| Input / condition | off TTFT | on TTFT | off pp | on pp |
|---|---:|---:|---:|---:|
| 6,308 tokens, repeated warm input | 13.810 s | 13.820 s | 456.77 | 456.43 |
| 12,934 tokens, repeated warm input | 28.086 s | 28.093 s | 460.52 | 460.39 |
| 27,964 tokens, unique records, cold Engram pages | 46.795 s | 42.481 s | 597.59 | 658.28 |

The cold-input median improves pp **10.16%**, reduces TTFT **9.22%** (4.31 s).
It is not a universal 10% gain: warm input is effectively unchanged.
The 577-output-token exact-copy decode fixture measures tg 45.265 → 45.222
(-0.09%, effectively unchanged). Short-output tg is not a decode benchmark.

Cold runs request `POSIX_FADV_DONTNEED` for each rank's two Engram shard files
before every request, after joining outstanding prefetch and resetting expert
cache. This is a controlled advisory page-cache eviction, not a claim about all
possible OS-cache states. The standalone long fixture is reproducible without
private Talk prompts using `tools/bench_engram_prefetch.py --long-only`.

For the cold fixture, demand gather/dequant total falls from 5.23/7.33 s on
head/worker to 2.33/2.99 s. Prefetch read work overlaps compute rather than
removing necessary reads. Measured total cgroup physical SSD reads are
109.30/110.74 GiB off versus 108.57/110.54 GiB on (includes expert reads and
minor routing/cache variation). No material extra SSD volume was observed.
The same output bytes pass in all 32 A/B requests (8 warmups + 24 measured).
Separate continuation/invalidation tests pass; the forecast and demand hashes
match for 8,023 subsequent prompt tokens on both ranks. Background read errors: 0.

Results: [summary](../results/engram-next-prefetch/summary.json),
[repeat-input runs](../results/engram-next-prefetch/comparison.json),
[cold-input runs](../results/engram-next-prefetch/long-comparison.json).
Private Talk fixture text remains outside the repository; stored results
contain hashes, metrics and fixed validation outputs only.

## Production validation

Standalone defaults and the embedded Talk recipe enable the feature; both
ranks were restarted with `DSV41_ENGRAM_NEXT_PREFETCH=1`, benchmark controls and
server development mode off. The rebuilt Talk arm64 application reconnects,
returns `spark-prefetch-ok`, emits pp/tg/ttft and deletes its temporary test
session. Head Extra x4 and worker Nemotron ASR are running again.

With those support services present, 54,988 tokens plus an 8192-token output
allowance pass the middle-marker check. The cold 27,964-token fixture also
passes in production (TTFT 41.724 s; different expert state, not the controlled
A/B result). Available memory stayed above 11.65/11.24 GiB on head/worker;
no OOM or reboot occurred. Both runtime file sets match.

A/B worker memory coverage was incomplete: the watchdog had treated Docker's
transient `created` state as a finished container. It now exits only for
`exited`/`dead`. The production validation has continuous records on both
nodes; the incomplete A/B worker record is explicitly marked missing rather
than reported as a memory minimum. Test watchers were stopped after validation;
normal serving and support services remain running.
