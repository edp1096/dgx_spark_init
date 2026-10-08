# Engram native buffered reader — qualified, 2026-09-13

Step 1 of the requested original-precision speed work passed controlled TP2
qualification. Native is now the serving default and is packaged in Talk.
Step 2 was rejected in [kernel qualification](COMPACT_MOE_QUALIFICATION.md);
step 3's [8192 window and dense tuning](PREFILL_WINDOW_8192.md) are also applied.

`engram_reader.py` uses persistent C pthread workers, one batched ctypes call,
and the same buffered positional reads as the original Python thread pool.
It neither changes the page-cache policy nor dequantization math. All submitted
workers finish before returning errors; fd/buffer lifetime stays with the caller.
Threads are bounded at 64. The current launch uses eight, as before.
`DSV41_ENGRAM_READER=python|native` selects the reader; default is `native`.
Set `python` explicitly to restore the previous read implementation.
The benchmark RPC is allowed only with `DSV41_BENCH_CONTROL=1`.

## Completed tests

- CPU integrity tests: mixed row widths/offsets, repeated and unordered rows,
  concurrent callers, truncated files, invalid fds, post-error reuse, empty
  requests, bounds and closed-reader behavior. All passed.
- Real checkpoint rows in both Engram tables on each physical node: equal BF16
  lookup outputs, masked rows and duplicates. No change to weights or scales.
- Six alternating paired timing trials per shape after warmup. The access pattern
  is synthetic (12 lookups/token/table), not a captured route trace.

| Lookup shape | Head Python → C | Worker Python → C |
|---|---:|---:|
| 1024-token equivalent | 216.41 → 24.62 ms | 190.97 → 23.41 ms |
| 4096-token equivalent | 916.29 → 98.99 ms | 932.74 → 90.80 ms |

These are CPU gather/dequantization timings, not model PP/TG speedups.
See `results/engram-reader/reader-rank{0,1}.json` and
`tools/run_engram_reader_check.sh`.

## TP2 qualification

`tools/bench_engram_tp2.py` alternates Python/native in one model process,
seeds the same 224 expert slots before every request, uses unique KV namespaces,
checks zero cached prompt tokens, exact expected output, byte-identical paired
responses and absence of concurrent inference. It includes the existing private
6308/12934-token tool fixtures and a longer exact-copy decode fixture.

The first pilot stopped before measurements because one worker's existing bind
mount still referred to the old gpu_worker.py inode. Files were synchronized
before recreating the next container. The next two launches failed during NCCL
initialization: `ibv_reg_mr_iova2: Cannot allocate memory`. No model measurements
were accepted. A model-free PyNccl all-reduce probe reproduced the failure;
`NCCL_CUMEM_HOST_ENABLE=0` did not fix it. Both nodes had >110GiB free RAM and
no compute processes after cleanup; no OOM or host reboot was observed.

Both hosts were on `7.0.0-1019-nvidia`. A matching report describes successful
kernel-only rollback tests on `6.17.0-1032-nvidia`:
https://forums.developer.nvidia.com/t/383023
The report was corroborating evidence. The older kernel, initrd, NVIDIA module
and GRUB entry already existed on both nodes; the user approved the reboot test.

Raw logs, source hashes and state are under
`~/.local/state/ds41-probes/20260913-speed`, with labelled external watchdog
records in `20260913-reader-ab`, `20260913-reader-ab2`, `20260913-reader-ab3`.

### Kernel-only follow-up

The user approved one-shot old-kernel boots. Worker then head rebooted into
`6.17.0-1032-nvidia`, keeping driver 580.173.02 and the same test images,
NCCL version/options and probe. The original model-free PyNccl test passed on
both nodes. No NCCL_CUMEM_HOST_ENABLE workaround was added. This supports a
kernel-dependent RDMA regression on these hosts. The boot selection was
one-shot; the persistent GRUB default was not changed.
After that test, the user authorized continued use of 6.17. Both nodes now have
a [persistent GRUB default](KERNEL_BOOT.md) selecting 6.17.0-1032; no further
reboot was needed to apply the default.
A worker-side supervisor then launched the guarded TP2 benchmark automatically
under probe token `20260913-reader-k617`. All 24 requests passed.

### Controlled results on 6.17

Three alternating timed pairs per fixture, excluding one warmup pair. These are
same-kernel reader comparisons, not comparisons against earlier 7.0 timings.
PP below is prompt tokens / TTFT, consistent with the earlier local reports;
it includes non-GEMM request work and is not a pure prefill-kernel throughput.

| Input | Python TTFT → native | Python pp → native | pp change |
|---|---:|---:|---:|
| 6,308 tokens, tool registry | 21.163 → 20.069 s | 298.06 → 314.31 tok/s | +5.45% |
| 12,934 tokens, document | 41.467 → 38.892 s | 311.91 → 332.56 tok/s | +6.62% |

TTFT medians fell 5.17% and 6.21%. The exact-copy fixture generated 577 tokens:
median tg was 45.52 → 46.08 tok/s (+1.23%), a small difference rather than a
demonstrated broad decode improvement. The two long-input fixtures output only
four tokens, so their tg numbers are not meaningful decode benchmarks.

All responses matched expected text, stopped normally, had zero cached prompt
tokens, and were byte-identical between readers. Both ranks used the same seeded
224-expert starting slots. Expert read counts varied slightly with execution,
so the trials do not enforce identical internal routing/cache traces. The native
reader itself is independently checked against identical row/scale bytes.
Warmup exclusion means these are warmed, controlled measurements, not a causal
first-request-after-cold-boot comparison.

Minimum available memory was 14.85 / 16.09 GiB; peak cgroup memory was 92.63 /
92.10 GiB. No OOM or watchdog trip occurred. Both kernels remained 6.17.
Evidence: `results/engram-reader/tp2-k617.json`, `tp2-summary.json`, and
`tp2-memory.json`. Full telemetry remains in private persistent probe state.
