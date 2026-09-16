# mmap demand reader comparison

> Cleanup 2026-09-13: rejected prototype source, experiment-only launch paths and dedicated replay tools were removed. The descriptions and commands below are historical records, not currently available runtime features. Adopted optimizations, safety guards and measurement evidence are retained.

Compare `DSV41_ENGRAM_READER=native` and `mmap`, keeping next-chunk prefetch
on in both variants. The latter continues to use its two-worker native pread
reader, so only demand row access changes.

The candidate uses PROT_READ/MAP_PRIVATE whole-file mappings and MADV_RANDOM.
The existing persistent native worker pool copies each requested byte range
with memcpy instead of pread. File extents and integer bounds are checked
before copying; checkpoint files must remain immutable while serving.
Same row ordering, dequantization and original weights. Mapping a file does
not make its whole contents resident. Mapping descriptors and page tables do
consume some memory, which is included in the monitored process lifetime.

Native-source C code remains byte-identical; a separate derived library is
compiled for mmap. Reader tests cover concurrent callers, remapping, exact
bytes, errors and truncation before a read. Server-side switching is available
only with benchmark controls enabled.

## Comparison protocol

One TP2 process, 224 target slots reseeded before each request, fresh KV salts,
identical requests, unchanged dense tactics and alternating variant order.
Three trials after excluding a warmup per fixture and variant. Forecasts are
joined before switching. Mappings are dropped before every sample, ensuring
that cold-cache eviction cannot leave mmap pages referenced by resident PTEs.
Consequently warm results describe warm **file pages** with mapping setup
included, not an ideal permanently-populated PTE cache. Cold runs also request
POSIX_FADV_DONTNEED on the two Engram shard files on each rank.

## Full-engine results

Three measured trials per variant, warmups excluded, next-chunk prefetch on:

| Fixture | Native TTFT | mmap TTFT | Native pp | mmap pp |
|---|---:|---:|---:|---:|
| 6,308 tokens / warm file cache | 13.845 s | 13.852 s | 455.62 | 455.39 |
| 12,934 tokens / warm file cache | 28.074 s | 28.122 s | 460.72 | 459.92 |
| 27,964 tokens / cold file cache | 42.520 s | 42.584 s | 657.67 | 656.68 |

The 577-output-token copy fixture measures tg 45.459 → 44.942 (-1.14%).
All 32 requests (8 warmups, 24 timed) return matching output bytes; read errors
are zero. This candidate provides no measured full-engine gain. The default
keeps buffered C reads; the follow-up below adds only current-row advice,
not mmap, to the default.

[Raw runs and summary](../results/mmap-engram/summary.json) retain the timing,
expert I/O and prefetch statistics. The warm mapping-reset caveat above limits
claims about an ideal permanently-mapped warm decode workload.


## Small-row WILLNEED follow-up

The plain mmap result does not test the recipe's small-gather WILLNEED prepass.
A separate component test sampled two tables × 72 unique rows/table (six-token
TP2 shape), eight alternating trials, original checkpoint bytes:

| Rank / state | Native | mmap | mmap + WILLNEED |
|---|---:|---:|---:|
| Head, warm | 0.159 ms | 0.302 ms | 0.486 ms |
| Worker, warm | 0.165 ms | 0.324 ms | 0.625 ms |
| Head, cold | 3.596 ms | 4.542 ms | 1.874 ms |
| Worker, cold | 3.175 ms | 3.264 ms | 1.444 ms |

This justified a separate full-engine decode test, rather than treating the
component latency as tg. Generate 1..200 exactly (400 output tokens, prompt
does not contain those output n-grams), reset expert cache and KV salt per
request, three trials after warmup per variant:

| File-cache state | native tg | mmap_hint tg | native_hint tg |
|---|---:|---:|---:|
| Warm repeated output | 40.982 | 40.921 | 40.953 |
| Cold before every request | 39.156 | 40.553 | 40.587 |

`native_hint` improves cold tg **3.65%**, warm tg changes −0.07% (effectively
unchanged). All 24 requests match the complete 400-token output exactly.
It retains the native pread worker pool and first issues POSIX_FADV_WILLNEED
for unique touched pages when the combined batch has at most 512 rows. Read batches above that limit bypass the prepass; a large repetitive
prefill can still qualify after row de-duplication. The subsequent pread remains authoritative;
unsupported optional hints do not suppress real read errors.

**Selected default: `DSV41_ENGRAM_READER=native_hint`.** Next-chunk prefetch
stays on. No mmap is used in production, and no decoded-row cache or expert
cache is added. Set reader to `native` for the previous plain C path. The two
hints serve different situations: next-chunk prefetch hides prompt I/O during
compute, while current-row advice batches pending decode page reads.

[Full-engine hint results](../results/mmap-engram/hints-summary.json).
These conditional gains must not be advertised as a universal tg improvement.

## Operational verification

Both ranks now run `native_hint`, next-chunk prefetch on and benchmark/dev
controls off. Original weights, KV 2 GiB/rank, context 65,536 and DCP1 remain.
Talk's embedded recipe and arm64 binary were rebuilt and the app restarted.
A 54,988-token middle-marker test, cold 400-token decode and real Talk marker
chat pass with head Extra x4 and worker ASR running. Talk emits pp/tg/ttft and
its temporary validation session is deleted. Observed minimum available memory
is 11.82/11.45 GiB (head/worker), without OOM or host reboot. Runtime source
hashes match across nodes. CPU reader/boundary tests: 14 pass; Go
orchestrator/config/server checks pass. The broader legacy GPU test modules
require PyTorch and were not run on the host Python; full-engine comparisons
above exercised the actual GPU runtime instead.
