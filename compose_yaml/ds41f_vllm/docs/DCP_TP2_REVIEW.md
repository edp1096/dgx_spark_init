# DCP2 review for the current TP2 streaming runtime

> Cleanup 2026-09-13: rejected prototype source, experiment-only launch paths and dedicated replay tools were removed. The descriptions and commands below are historical records, not currently available runtime features. Adopted optimizations, safety guards and measurement evidence are retained.

Scope: feasibility and cost review, not a claim that DCP serving is implemented.
The serving configuration remains DCP1. Original weights, next-chunk Engram
prefetch and the current expert cache are retained.

## Actual cache layout

A read-only benchmark RPC captured each rank's live KVCacheConfig and specs.
Both ranks have one aliased backing allocation of 2,147,420,160 bytes, not twenty
independent 2 GiB tensors: tensor entries overlap the same pool. There are
15,534 blocks, each 138,240 bytes, and eighteen cache groups:

- Fifteen replicated SWA groups cover forty target and three draft layers.
- Two full-attention groups hold four compressed-KV sources and their four
  indexer K caches (layers 2/8/14 at compression ratio 2, layer 20 at ratio 1).
- One replicated compressor-state group holds three circular buffers.

At context 65,536 and the existing 8192 scheduler/in-flight allowance:

| Admission calculation per rank | Current | Estimated DCP2 |
|---|---:|---:|
| Replicated SWA blocks | 3,885 | 3,885 |
| Replicated circular-state blocks | 1 | 1 |
| Shardable compressed/indexer blocks | 1,536 | 768 |
| Total admission blocks | 5,422 | 4,654 |
| Padded pool bytes for these blocks | 714.81 MiB | 613.56 MiB |

This saves **101.25 MiB/rank of single-request admission capacity** under the
current grouping/padding model. The shardable content occupies 112.5 MiB;
halving that raw storage saves 56.25 MiB before common-pool padding. The current
fixed 2 GiB allocation does not become smaller automatically. An unchanged
pool would have an estimated capacity-equivalent 218,744 vs reported 187,760
tokens (+16.5%), not twice the capacity. These are allocator estimates, not
validated context limits or concurrent-request settings. Configured context
stays 65,536 and max_num_seqs stays one. DCP workspace and prefix-retention
behavior could reduce the predicted benefit.

For scale, one additional expert slot in all forty target layers requires
358.59 MiB/rank (9,400,320 bytes per expert × 40). The estimated single-request
saving is insufficient for even that uniform cache increment. It cannot make
all experts resident.

## Required code changes

1. Per-group replicated/sharded flags through cache specs, group allocation,
   block tables, slot mapping, admission and prefix-cache reuse. Current
   SlidingWindowSpec explicitly rejects DCP > 1.
2. Compressed-state position mapping and global top-k merging. The current
   candidate-block indexer also explicitly rejects DCP > 1. The checkpoint
   uses candidate blocks of eight, so upstream examples using four must not
   be copied literally.
3. Decode query gather and attention output/LSE merge, including correct
   treatment of replicated window slots, sink probability and empty shards.
   TP2 has 32 local query heads, so DCP2 processes 64 gathered heads.
4. Prefill compressed-KV gathering and global-to-gathered index mapping, to
   avoid exchanging the much larger query/output tensors at every layer.
5. Preserve window-only DSpark draft caches and their input mapping. Keep
   target intermediate states needed by DSpark intact.
6. Validate odd sequence lengths, chunk boundaries, prefix reuse, cancellation,
   long context, speculative acceptance, output scores and memory peaks.

The installed SM120 low-level sparse MLA implementation exposes output LSE
and supports a decode envelope through 128 heads. Those are useful primitives;
they do not supply the missing V4.1 cache/scheduler integration above.

## Communication dimensions

The original checkpoint has 38 compressed-attention layers, 32 local query
heads at TP2, head_dim 512 and DSpark target verification sizes around 5–6.
Each DCP layer needs a query gather and an output/LSE exchange. At six tokens,
the query payload is 196,608 bytes/rank, and a BF16-output/FP32-LSE remote
payload is 197,376 bytes/rank (393,984 if output is packed as FP32).

At 8192 prefill tokens those query/output messages are approximately
256/257 MiB **per layer**. Gathering only remote compressed KV at 65K context
instead transfers about 45.70 MiB across the four distinct KV sources, plus
indexer communication and buffer/index transformations. Our ratios 2/1 imply
163,840 compressed states across those sources at 65K; a quoted generic
'few thousand compressed states' must not be used for this checkpoint.

Transport measurements are recorded separately. They measure communication
only, not a DCP engine, attention correctness, merge kernels or model speed.

References: [live cache audit](../results/dcp-review/kv-audit.json),
[allocation estimate](../results/dcp-review/kv-estimate.json),
[AidenLab design](https://aidenle.com/recipes/deepseek-v4-1-flash-4x-dgx-spark/).

A full compressed-KV gathered buffer at 65K would itself occupy about 91.4 MiB
per rank if all four sources were retained for a step. Reusing buffers between
source groups could lower the peak, but indexer/LSE workspaces must also fit.
Thus the 101.25 MiB admission saving is not a net host-memory saving promise.
There is little room to turn this into faster MoE execution in the current
65K, single-request, SSD-expert-streaming workload.

## Measured transport and decision

Model-free PyNccl 2.30.7 on the same two GB10 nodes, 6.17 kernel, same IB/GID/NIC
settings, original serving image; 60 samples per message size after warmup.
Every gather/exchange passes exact payload checks.

- Six-token query gather: 51.17 µs; remote BF16-output/LSE-sized exchange:
  58.16 µs. Across 38 layers, approximately **4.15 ms per target verification
  step**, before packing, LSE merging, indexer work or attention changes.
  FP32 output packing raises this estimate to 4.43 ms.
- Naive 8K-prefill query gather and output exchange: 24.96 + 21.06 ms per
  layer, approximately **1.75 s per prefill chunk** across 38 layers.
- Gathering only remote compressed KV from four sources at 65K: **4.92 ms**
  total transport, before indexing/copies. That optimized path is necessary
  if a DCP port is pursued; it is not present in our current V4.1 attention.

These are transport measurements and structural extrapolations, not a measured
DCP model throughput regression. The full port could overlap some work and
would also add work not measured here. Nevertheless, current speed-focused
work has no compelling DCP payoff: only ~101 MiB/rank of single-request block
savings, new workspaces, and extra decode communication. **Do not enable/port
DCP for this current 65K single-request speed target.** Revisit if longer
contexts or more concurrent contexts become the objective.

[Transport samples](../results/dcp-review/transport.json).
