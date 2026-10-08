# io_uring and native Expert callback experiments — 2026-09-13

> Cleanup 2026-09-13: rejected prototype source, experiment-only launch paths and dedicated replay tools were removed. The descriptions and commands below are historical records, not currently available runtime features. Adopted optimizations, safety guards and measurement evidence are retained.

**Decision: keep `native_hint`, next-chunk Engram prefetch, eager whole-model
execution and existing individual Expert CUDA graphs.** Neither candidate is
enabled by default. The native callback did not pass the performance gate for
a full-model port. This is not a claim that every possible implementation of
these techniques will be slower.

## io_uring: real TP2 inference

The experimental reader preserves original Engram weight and scale bytes. It
deduplicates 4 KiB pages, coalesces adjacent pages into at most 64 KiB reads,
uses up to 64 registered staging buffers (4 MiB) and scatters bytes back into
the original row order. It uses buffered file descriptors, retaining the OS
cache and the current small-batch WILLNEED hint. There is no new persistent
Engram row cache, quantization or extra Expert allocation.

Liburing 2.9 is pinned to `08468cc3830185c75f9e7edefd88aa01e5c2f8ab`.
The production container's default seccomp policy rejects io_uring setup.
The temporary experiment used a pinned Moby default profile with only the
three io_uring syscalls additionally allowed. It did not change Docker daemon
policy or install system packages.

Both readers ran in the same TP2 process: original checkpoint, DSpark5,
224/128 Expert slots, KV 2 GiB/rank, context 65536, scheduler 8192, kernel
2048, shared buffers and 384 MiB scratch. Dense prefill tactics, next-chunk
Engram prefetch and Expert I/O overlap remained enabled. Each sample seeded
the same Expert cache and used a fresh KV salt. Cold tests used advisory
eviction of the two Engram files after draining prefetch. Cold eviction is
not a guarantee that every OS/device cache is empty.

Three timed trials per variant, alternating order, plus excluded warmups:

| Workload | native_hint | io_uring + hint | Change |
|---|---:|---:|---:|
| Warm 6,308-input TTFT | 13.833 s | 13.921 s | pp −0.63% |
| Warm 12,934-input TTFT | 28.158 s | 28.291 s | pp −0.47% |
| Cold 27,964-input TTFT | 42.457 s | 41.741 s | pp +1.71% |
| Repeated copy, 577 output tokens, tg | 45.306 | 45.835 | +1.17% |
| Warm generation, 400 output tokens, tg | 40.519 | 40.132 | −0.96% |
| Cold generation, 400 output tokens, tg | 40.035 | 41.233 | +2.99% |

The short marker fixtures are used to evaluate prefill, not meaningful decode
throughput; their few-token tg figures are not acceptance evidence. The 400-token
fixture generates integers 1 through 200, absent from its prompt. The copy
fixture intentionally exercises a different, highly predictable workload.

All 48 requests matched expected output, including 36 timed samples and 12
warmups. There were no guard trips or host reboots. Minimum available memory
was 11.34/12.45 GiB on head/worker; maximum sampled cgroup use was 91.82/92.04
GiB under the 100 GiB hard limits. CPU integrity tests cover concurrent callers,
mixed row widths, invalid fds, short reads, draining, reuse and lifetime.

Component cold-read improvements were much larger than end-to-end gains:
4,096 random rows/table with queue depth 64 took about 84/81 ms, versus
155/152 ms for the original reader. Warm component reads regressed from about
4.6 ms to 17/16 ms. The full-model results, rather than those component numbers,
drive the decision to retain the existing default.

`tools/prepare_uring_probe.py` reconstructs temporary dependencies and the
scoped seccomp file. `DSV41_URING_PROBE=1` requires benchmark controls; the
`uring_hint` backend is explicitly unqualified for normal serving. CPU tests:

```bash
python3 tools/prepare_uring_probe.py
python3 tools/test_engram_uring.py
```

The final experimental reader aborts on submission/completion-queue infrastructure
failure rather than risk releasing buffers still owned by outstanding I/O.
Ordinary fd/EOF failures still drain and report errors. This error-path-only
hardening followed timing; integrity tests were rerun. Timed and final source
hashes are recorded separately. Queue-infrastructure fault injection was not
performed, and the reader remains a benchmark prototype.

## Native Expert callback and CUDA Graph: isolated GPU gate

`expert_callback.c` performs slot lookup, requested-set protection, LRU eviction
and exact packed-file reads in native code. A graph orders route-ID D2H, the CPU
callback, mapped-ID H2D and the original b12x kernels. Four persistent CPU readers
write prevalidated CPU aliases of the existing registered weight allocations.
The callback makes no CUDA API calls. Pointer discovery happens before capture
through `expert_pointer.cpp`, compiled against installed CUDA headers.

The comparison uses actual TP2 packed layer-0 weights on each server, 384 experts,
224 cache slots and 9,400,320 bytes/expert/rank. Both sides use the current
b12x numerical recipe, shared buffers and the same kernels. The baseline is
the existing Python slot manager plus individual Expert graphs and batch I/O.
Target shapes are one token and six tokens. The test does not include attention,
shared Expert calculation, TP communication or whole-model scheduling.

Production inference was stopped. Each experiment container had an 8 GiB hard
limit, no swap allowance and an external 10 GiB host-free guard. There were no
OOMs or host reboots; successful component runs retained at least 113.44 GiB
available. Three timed trials and excluded warmups were used per variant.

Six-token results, milliseconds per layer (lower is better):

| Cache condition | Head baseline → callback | Worker baseline → callback |
|---|---:|---:|
| All hits | 2.108 → 2.582 | 2.120 → 2.722 |
| One new Expert per call | 4.104 → 3.730 | 4.162 → 4.767 |
| Four new Experts per call | 7.623 → 7.216 | 7.791 → 7.849 |
| Heavy eviction | 47.070 → 46.351 | 44.825 → 44.653 |

One-token all-hit execution also regressed: 0.429 → 1.027 ms on the head and
0.423 → 0.563 ms on the worker. Native all-hit callback execution itself was
below 1 microsecond on the head. The larger end-to-end cost is outside that
cache computation: graph/host handoff, copies and scheduling are included, but
their individual contributions were not separately profiled. CPU slot lookup
speed alone does not establish a graph speedup.

All 384 untimed GPU output comparisons were bit-exact. Final used slots matched
packed checkpoint byte hashes, and callback miss/read-byte totals equalled the
baseline for each validation sequence. Mixed-miss cases were added because hit
and eviction extremes alone do not describe normal serving. Their results still
did not establish a consistent win on both TP ranks. No whole-model callback
speed or memory claim is made, and no whole-model callback port was enabled.

Reproduce only after stopping that node's production model:

```bash
python3 tools/test_expert_callback.py
python3 tools/run_callback_component.py --rank 0
python3 tools/run_callback_component.py --rank 0 --suite mixed
```

Use rank 1 on the worker. The runner refuses a running production rank, records
memory and container logs, and kills only its own immutable container ID if its
guard trips. Remove its stopped, labelled experiment containers before repeating.

### Harness failures excluded from timings

The initial ctypes mirror of CUDA pointer attributes omitted CUDA 13's reserved
fields. The query overwrote Python memory and both isolated processes exited 139
during baseline kernel preparation, before callback execution. The setup-only
C++ helper removed this duplicated ABI. A later six-token synthetic eviction
trace incorrectly assumed 512 experts; the existing baseline bounds check rejected
it. Trace generation now uses the packed header's actual 384-expert count.
Complete successful measurements above were collected after these fixes. Failure
logs and partial diagnostic results are retained separately, not averaged in.

## Evidence and restoration

Raw results, source hashes, summaries, failed-harness diagnostics and the restored
service receipt are under `../results/uring-graphs/`. In particular:

- `uring-summary.json`, `model-warm.json`, `model-cold-long.json`,
  `model-decode-warm.json`, `model-decode-cold.json`.
- `callback-rank{0,1}.json`, `callback-mixed-rank{0,1}.json`.
- Model and component memory receipts, `source-hashes.json`,
  `final-source-hashes.json`, and `restored.json`.

Restoration uses normal container policy, `native_hint`, Engram next-chunk
prefetch, context 65536, KV 2 GiB/rank, model graphs off and Expert graphs on.
Benchmark RPC/dev mode is disabled. ASR, all four Extra services and Talk are
checked alongside a real model response.
