# Faster uncached prefills with bounded expert execution

This is the earlier scheduler/grouping experiment. The subsequent native
2,048-token kernels and shared graph I/O are documented in
[PREFILL_ARCHITECTURE.md](PREFILL_ARCHITECTURE.md). The subsequent qualified
4,096 scheduler is documented in [PREFILL_4096_RETRY.md](PREFILL_4096_RETRY.md).

The SparkTalk integration processes about 6,300 input tokens even for a short
question when its actual system instructions, recalled context and tools are
included. With a 512-token scheduling budget, a request traverses the target
layers in roughly thirteen chunks. Missing experts can be read again on later
chunks. Prefix reuse helps subsequent requests but does not accelerate the
first processing of these tokens.

This experiment increases the scheduler budget to 1,024 and 2,048 tokens while
keeping each b12x execution at at most 512 tokens. `SlotLayer._run` selects and
loads an expert group for the whole scheduled prefill. `execute` then computes
successive token slices against those resident experts. It copies each slice
out of the reusable graph buffer before the next replay overwrites that buffer.
Only after all slices are complete can the next expert group replace the slots.

The initial 1,024-token pilot reduced the short fixture's timed TTFT only to
63.65 s (99.18 tok/s), versus the 512 baseline mean 66.74 s. This pilot had one
timed short run and is not a completed suite. Its following document request
was deliberately cancelled before refining the expert grouping policy. The
partial record is `results/pp-batch1024-tiles-pilot.json`.

The final grouping policy first consumes the needed experts already present
in the slot cache, then loads the missing experts in capacity-sized groups.
The previous frequency-only ordering could evict a resident expert while
loading the first group, then reload that expert for a later group. Consuming
resident experts first eliminates this avoidable read. Groups retain the
frequency ordering within each resident/missing set. FP32 partial sums can
therefore differ by rounding, while checkpoint/activation precision is intact.

This keeps the original weight precision, V4.1 activation quantization, router
weights and fixed b12x execution plans. Expert slots stay uniform224 and the
shared scratch arena stays 256 MiB per stream. Dense/attention activations and
the combined output still grow with the scheduler budget; the launcher adds
one GiB of required host headroom per extra 512 scheduled tokens. Larger
prefills are restricted to eager whole-model execution and at most 2,048 tokens.

The general scheduling direction is also described in the
[vLLM tuning documentation](https://docs.vllm.ai/en/stable/configuration/optimization/#performance-tuning-with-chunked-prefill).
This custom SSD-streamed model requires its own tests; the documentation is
not evidence of a measured improvement on this machine.

## Numerical and I/O checks

`tools/run_prefill_tile_check.sh` runs `tools/check_prefill_tiles.py` independently on each
rank using actual packed checkpoint experts and a 24-slot cache. The cases are
1,024 tokens with 12 resident experts, and 1,537/2,048 tokens with 48 experts
that require eviction. They compare the new larger call with the qualified
512-token call sequence, include a partial final tile, verify that the returned
output survives the next graph replay, and assert exactly one read per required
expert for the whole larger call.

Both ranks passed. The resident case was bit-identical; the two eviction cases
had a maximum absolute difference of 1.862645149230957e-9. Scratch remained
268,435,456 bytes. The small FP32 summation difference is not extra quantization.
Evidence: `results/prefill-tiles-rank{0,1}.{json,log}`.

After adding resident-first grouping, the reference was frozen to the original
frequency-only 512-token algorithm so the test does not compare two copies of
the new grouping logic. A 512-token eviction case and a second call with useful
cached experts were added. Both ranks passed all four sizes; the worst absolute
difference was 3.725290298461914e-9. With 24 of 48 needed experts already cached,
the second call read only the 24 missing experts (225,607,680 bytes). The fully
resident case issued zero reads. Evidence: `results/prefill-resident-rank{0,1}.{json,log}`.

## Benchmark conditions

`tools/bench_prefill.py` accepts private request fixtures captured through a temporary
SparkTalk instance using a copy of the real database and settings. The instance
uses a stub model endpoint only to capture request construction. The live user
database and configuration are not changed. Both fixtures include the actual
14 tools. One is a short marker question; the other adds mixed technical prose
and code from this recipe's README. The frozen requests are identical across
the scheduler settings, including all reference context.

For each setting, each fixture has an excluded warmup followed by two timed
runs. Before every request, the opt-in benchmark control resets both ranks'
expert slot mappings and I/O counters. Every request has a unique `cache_salt`,
and the result must report zero cached prompt tokens. Compiled kernels and
allocated buffers remain warm in timed runs, so these measure uncached input
processing rather than the entire model startup. No global OS cache is dropped.
O_DIRECT expert transport, batch_overlap, original precision, DSpark5, 65,536
context, 2 GiB KV and prefix retention128 remain the same.

The timing fixtures disable reasoning and allow up to 16 output tokens to
isolate input processing. Their answers must match the specified marker and
stop normally. This does not alter the running application's reasoning or
8,192-token output allowance. `pp` uses prompt tokens divided by the engine's
scheduled-to-first-token time; `ttft` additionally includes queue time.
Periodic expert log counters are supporting evidence and may stop slightly
before the final model step; they are not exact per-request final counters.

The 512 baseline uses the original grouping; final candidates combine resident
grouping with 1,024/2,048 scheduler budgets. Results use labels `pp-batch512`,
`pp-batch1024-resident` and `pp-batch2048-resident`, with per-request rank logs. Raw
fixtures contain private context and stay outside the repository; result files
contain request hashes and measurements only.

## Measured comparison

Throughput is weighted by input tokens / input-processing seconds over two
timed runs per fixture. TTFT is the median of those two runs. Every run reported
zero prompt cache hits and returned the correct marker.

| Setting | 6,313-token pp | TTFT | 12,270-token pp | TTFT |
| --- | ---: | ---: | ---: | ---: |
| Original 512 | 94.58 tok/s | 66.74 s | 97.85 tok/s | 125.39 s |
| Resident-first 1,024 | 121.84 tok/s | 51.81 s | 134.81 tok/s | 91.02 s |
| Resident-first 2,048 | 144.59 tok/s | 43.66 s | 171.46 tok/s | 71.56 s |

The 1,024 candidate improves pp by 28.8% / 37.8%, with latency reductions of
22.4% / 27.4%. Last periodic expert read counters fall from about 340.4 to
250.5 GiB per rank for the shorter input, and 632.2 to 425.4 GiB for the
document input. These counts support the reduction in redundant SSD reads.
The smallest post-request host availability in those candidate trials was
14.47 GiB. They do not measure arbitrary workloads or a universal optimum.

`tools/compare_prefill.py` checks identical request hashes and token counts before
producing `results/pp-batch-comparison.json`.

The selected 2,048 default improves pp by **52.9% / 75.2%** over the original
512 setting. TTFT falls **34.6% / 42.9%**. Periodic reads are about 201.8 /
309.0 GiB per rank, compared with the original 340.4 / 632.2 GiB. The minimum
post-request available host memory across its trials is 13.31 GiB. All 18
requests across the three completed suites, including their excluded warmups,
return the exact expected marker with zero prompt-cache reuse.

The production default is 2,048 only for `b12x_slots`; diagnostic backends keep
512. `DSV41_MAX_BATCHED_TOKENS` can override the size on both ranks. Model graph
experiments must explicitly use 512. These gains combine larger scheduling
chunks and resident-first expert grouping; they are not an isolated GPU GEMM
speedup, and they do not rely on precomputing the user's prompt at startup.

## Generation checks

`tools/smoke.py --suite long --repeat 2` generated a binary-search function and a
Rayleigh-scattering explanation, each with an immediate repeat, before and
after the change. All four texts were byte-identical and stopped normally.
The baseline function passed found/missing/empty-list behavior checks; the
candidate emits the identical function. Weighted decode throughput was
17.85 versus 17.68 tok/s (-0.9%) in this small check. This does not establish
a general generation-speed change. Evidence: `results/pp-batch512-decode.json`,
`results/pp-batch2048-decode.json` and `results/pp-decode-comparison.json`.

## Production deployment validation

Both ranks were restarted using the new 2,048 default and
`DSV41_BENCH_CONTROL=0`; the temporary control file was removed. No startup
prompt was submitted to prime the user's shared prefix. The running SparkTalk
instance then processed two actual chat turns with its normal 14 tools,
reasoning `max`, 65,536 context and 8,192-token output allowance.

| Actual application request | Input / cached tokens | pp | ttft | First event received | Completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| First after restart | 6,333 / 0 | 133.51 tok/s | 47.44 s | 47.49 s | 62.04 s |
| Immediate follow-up | 6,362 / 6,144 | 69.45 tok/s | 8.26 s | 8.63 s | 16.48 s |

Both marker answers were exact, persisted measurements matched the final SSE
events, and the temporary session was deleted. The first application request
includes first-use work that the warmed-kernel PP benchmark excludes. The
follow-up pp counts only 218 new tokens and its ttft includes queue waiting;
it is not an uncached PP comparison. Total completion time also includes
reasoning/output generation. Evidence: `results/sparktalk-pp-validation.json`.

The subsequent 54,988-token capacity request retained the 8,192-token output
allowance, retrieved the exact middle marker and stopped normally. First
content arrived at 50.24 s. Its repetitive prose is a context-capacity check,
not a general-document throughput benchmark. A following short request with
the same output allowance returned exactly `OK` in 0.56 s. Evidence:
`results/pp-context-capacity-validation.json` and `results/pp-runtime-validation.json`.

Final runtime inspection confirmed both ranks healthy, batch2,048,
retention128, KV2 GiB, maxseq1, benchmark control off, expert tensor payload
87,836,590,080 bytes and scratch268,435,456 bytes. Source hashes match across
the hosts. After the long input, host availability was 9.84/10.11 GiB; neither
container was OOM-killed. The runtime JSON records cgroup current/peak/limit
values and source hashes. Larger prefills need extra activation memory even
though expert slots and the shared expert-kernel scratch allocation stay fixed.

Private capture DB/config/log files were removed. The frozen request fixture
remains outside the repository in the owner's mode-0700 temporary directory,
with file mode0600, for reproducing the comparison without publishing recalled
context. User model weights, memory records and normal chat history were not
modified by this tuning.
