# Shared expert execution and actual-route staging

This records round 4. The later monitored 4096 retry succeeded and supersedes
the scheduler cap/default and interruption status below; see
[PREFILL_4096_RETRY.md](PREFILL_4096_RETRY.md). Its paired measurements use new
private fixtures because the earlier ones were lost on reboot.

This round starts from resident-first grouping with a 2,048-token scheduler,
512-token expert kernels, 224 target slots per layer and 128 draft slots. It
keeps the original MXFP4/E8M0 checkpoint and V4.1 MXFP8 activation contract.

## Shared graph I/O

Each CUDA stream owns one input, route-ID, route-weight and output buffer per
(token capacity, top-k) pair. Layer-specific execution plans and CUDA graphs
bind views of these shared tensors. Different streams have separate storage.
The graph output is borrowed: its consumer must enqueue its copy or arithmetic
on that stream before another layer uses the same capacity. The serving caller
consumes it in the shared-expert addition and all-reduce before the next layer.
The large tiled path copies every partial output before replaying the next tile.

Scratch remains a fixed arena per stream. The pinned b12x planner requests
146,936,028 bytes at 512 tokens, 199,076,380 at 1,024 and 303,367,324 at 2,048
with 224 experts and top-k 6. Native 2,048 uses a 384 MiB arena. No captured
arena is resized. Scheduler capacity and expert kernel capacity are independent.

`tools/check_shared_prefill.py` compares real packed weights against the frozen
pre-change implementation in `results/pp-round4-reference.py`. Both ranks
passed 72 comparisons across target layers 0/1 and draft layer 40, top-k 6/3,
6/511/513/1024/1537/2048/3073/4096 tokens and all three kernel limits. It checks cold read
counts, partial padding, consumed output lifetime and separate CUDA streams.
The largest absolute difference was 5.960464477539063e-8. This is FP32
summation rounding, not additional weight quantization.

A resident-only 2,048-token execution microbenchmark with 24 experts measured:

| Rank | 512-token tiles | Native 1,024 | Native 2,048 |
| --- | ---: | ---: | ---: |
| 0 | 8.965 ms | 7.713 ms | 6.692 ms |
| 1 | 9.110 ms | 7.466 ms | 6.438 ms |

These are isolated kernel-path timings, not end-to-end pp improvements.

## Actual-route staging candidate

`routed_pipeline.py` allocates a separate registered bank of 160 experts per
CUDA stream, shared across target layers. With uniform 224-slot target caches,
the 384 possible expert IDs fit in one resident group and one staging group.
All already-needed resident experts are protected while filling the first
group. The remaining actual routed IDs are read with O_DIRECT into the staging
bank while the resident group computes. No forecast IDs or extra expert reads
are introduced. After staging compute, weights are copied into evictable LRU
slots so subsequent chunks and decode can reuse them.

The separate bank prevents any concurrent reader from overwriting weights or
scales still in use by a resident kernel. CUDA graph captures are primed before
reader submission because a global capture cannot overlap native stream fences.
Every submitted read is joined even if the shared-expert callback raises.
Consumers and cache-retention copies run before the next layer can reuse the
bank. Different CUDA streams get distinct banks.

The bank costs 160 × 9,400,320 = 1,504,051,200 bytes (1.401 GiB) per rank and
stream, plus graph plans. It also adds GPU copies into the persistent cache;
only full-request measurements can establish whether overlapping the reads
outweighs this cost. It is restricted to eager model execution, native batch
I/O, shared graph I/O and a uniform cache without predicted prefetch.

## Measurement protocol

The same two frozen real SparkTalk requests (6,313 and 12,270 prompt tokens,
14 tools) are used throughout. Raw fixtures remain private outside the repo.
Each setting gets one excluded warmup and two timed requests per fixture.
Every request resets expert mappings on both ranks and uses a unique KV cache
namespace; all measured responses must have zero cached prompt tokens, the
correct marker and normal stop. No global OS cache is dropped. Reasoning is
disabled only in the measurement requests, which allow 16 output tokens.
Normal SparkTalk retains its reasoning setting and 8,192-token output allowance.

Buffer variants are compared within one server boot. Allocations from earlier
variants remain resident in that boot, so their post-request free-memory values
are not an isolated estimate of the candidate's memory savings. Final normal
serving memory must be measured after a clean restart. Periodic expert counters
are snapshots, not exact final request totals.

## New b12x commit qualification

A separate image installs upstream commit
`081b235931dbbcedcf0eb5899bae990c5dec5238` over the pinned runtime with no
dependency upgrades. Both ranks independently repack source experts 0 and 17
from a target and draft layer. Shapes, dtypes and every packed byte match the
existing files. For 1/5/6/16/512/2048-token executions, all measured output
tensors were bit-identical to the old image.

The new kernels nevertheless regress this TP2 shape. Rank 0 target execution
latencies were 0.623 → 1.482 ms at five tokens, 0.797 → 1.752 ms at six tokens,
and 6.568 → 9.135 ms at 2,048 tokens. Rank 1 independently showed the same
regressions (0.630 → 1.468, 0.761 → 1.730 and 6.242 → 8.574 ms). One-token
execution improves, but the dominant DSpark5 verify/draft shapes and prefill
regress. The package replacement is rejected before full-model deployment;
these are expert-path measurements, not measured whole-model tg regressions.
The production image remains `dgx-ds41-stream:b12x8`, commit `789bbb3c`.

Evidence: `results/pp-round4-upgrade-{base,new}-check-rank{0,1}.json` and
`tools/check_b12x_upgrade.py`. Reference tensors stay outside the repository in the
local test cache. Reproduce with `bash tools/run_architecture_check.sh RANK upgrade-base`
then `upgrade-new` while the serving model is stopped.

Additional shared-buffer tests passed 72 comparisons per rank, extending the
input sizes to 3,073 and 4,096 tokens. The routed pipeline passed cold and warm
256/384-expert routing cases, exactly-once missing reads, callback-failure
recovery, separate CUDA streams and cross-layer bank reuse. The final retention
copy uses four indexed GPU copies instead of one copy per expert plane.

## Full-request results and selected defaults

Two timed requests per fixture, with excluded warmups and zero prefix hits:

| Setting | 6,313-token pp | TTFT | 12,270-token pp | TTFT |
| --- | ---: | ---: | ---: | ---: |
| Rechecked previous implementation | 139.49 tok/s | 45.26 s | 171.20 tok/s | 71.67 s |
| Shared I/O, 512-token kernels | 144.25 tok/s | 43.77 s | 171.31 tok/s | 71.63 s |
| **Shared I/O, native 2,048 (selected)** | **158.29 tok/s** | **39.88 s** | **193.22 tok/s** | **63.50 s** |
| Native 2,048 plus routed staging | 160.60 tok/s | 39.31 s | 195.60 tok/s | 62.73 s |

The old short-input control varied from 43.83 to 46.68 seconds. Shared I/O alone
has no established speed benefit: its short result matches the faster control
trial and its long result is unchanged. Comparing native kernels with the
shared-I/O 512 control isolates +9.74% / +12.79% pp and saves 3.88 / 8.12 seconds.
The preceding round's old-kernel result was 144.59 / 171.46 tok/s, consistent
with that conservative comparison. New native-kernel timed ranges were
157.93–158.65 and 192.87–193.58 tok/s.

Staging adds only 1.46% / 1.23% beyond native 2,048, while reserving another
1.401 GiB per rank and adding retention copies and initial graph setup.
It remains implemented and tested behind `DSV41_ROUTED_PIPELINE=1`, but the
serving default is off. Its optional launcher requires uniform 224 slots,
native batch overlap, shared buffers and eager model execution.

Normal defaults are scheduler 2,048, kernel 2,048, shared graph I/O on,
384 MiB scratch, routed staging off, benchmark control off, unchanged uniform
224/128 expert slots, DSpark5, 65,536 context and 2 GiB KV. The launcher sums
extra KV, scratch and staging byte allocations before one GiB rounding step;
it retains the existing base headroom and scheduler allowance. It never drops
global caches or stops unrelated jobs to satisfy the guard.

## Interrupted 4,096 startup and recovery

At 16:28:19 KST on 2026-09-12 the head started the 4,096 scheduler candidate.
At 16:29 the head host rebooted while the model was still starting. No health
success or benchmark response was recorded. The worker host remained up; its
model worker waited for the missing head. Persisted head kernel logs contain
no OOM, NVIDIA Xid or panic report, and pstore was empty on the new boot.
Docker reports exit 255 and OOMKilled=false. These facts establish a host reset,
but **do not establish its cause or exclude a memory/driver failure**. The
4,096 trial is unqualified and was not retried. Its startup log, filtered
container state and prior-boot journal are saved in `results/pp-round4-*`.

The launcher again rejects scheduler capacities above 2,048. The surviving
worker process was stopped; both ranks were restarted with the selected
2,048 configuration and benchmark controls removed. SparkTalk was restarted
from the existing binary and configuration; other model services remain down.
The reboot cleared the private fixtures under `/tmp`. Their request hashes,
token counts, answers and completed measurements survive in results, but the
exact frozen request bodies must not be claimed to remain reproducible.
A future comparison needs a new private fixture capture and matching baseline.

## Final normal-serving validation

The selected configuration was restarted with benchmark control off and no
benchmark control files. The first inference after health was a real SparkTalk
request with 14 tools, reasoning effort `max`, 65,536 context and 8,192 output
allowance. It returned the exact requested marker; the second turn correctly
replaced that marker. Stored performance metrics match SSE and only the
validator's temporary session was deleted.

| Actual SparkTalk turn | Prompt / cached | pp | tg | TTFT |
| --- | ---: | ---: | ---: | ---: |
| First after restart | 6,333 / 0 | 146.92 | 19.60 | 43.10 s |
| Follow-up | 6,362 / 6,144 | 80.40 | 18.93 | 7.67 s |

The prior actual first turn was 47.44 seconds; this run was 43.10 seconds.
Actual reasoning and first-use overhead differ from the warmed-kernel fixture
benchmark. Prefix hits can reduce waiting time without proportionally increasing
the displayed pp, which measures uncached tokens.

A 54,988-token input with an 8,192-token output allowance returned its exact
middle marker with HTTP 200 and normal stop, followed by a successful short
request. This repetitive long input checks capacity, not general pp throughput.
Afterward available RAM was 14.19 / 11.37 GiB. Head/worker cgroup peaks were
99,200,335,872 / 96,236,244,992 bytes, below the 107,374,182,400-byte limits;
OOM and OOM-kill events remained zero. Per rank, expert tensors remain
87,836,590,080 bytes, graph I/O uses 75,689,160 bytes, scratch is 402,653,184
bytes and no staging bank is allocated. Host reboot/background-state changes
mean free-RAM differences from older runs are not a measured buffer-savings
estimate.

The four deterministic code/prose decode responses match the earlier validated
responses byte for byte and all stop normally. Weighted tg was 18.61 tok/s
versus 17.68 previously. Different initial expert-cache states and this small
sample make it a regression check, not a general 5.2% decode-speedup claim.
Evidence: `pp-round4-sparktalk`, `pp-round4-context`, `pp-round4-runtime`,
`pp-round4-decode` and `pp-round4-decode-comparison` in `results/`.
