# Cache preload, phase separation and decoder replay review

Reviewed 2026-09-12 against upstream commit
[`8b68fdde188fea5e8e9cd030f0e1b299dc16ac85`](https://github.com/0xBakeer/deepseek-v41-flash-spark/tree/8b68fdde188fea5e8e9cd030f0e1b299dc16ac85).
No serving flags, weights or running containers were changed for this review.

## 1. Frequency-based preload

Applicable without quantization or changing routing. Our SlotLayer allocates fixed
224 target slots per layer and starts its mapping empty; demand reads populate it.
There is no persistent workload-frequency seed. The existing optional predictor
is a different feature (forecast upcoming selections during execution) and remains OFF.

A seed should be learned from separate prior requests, keyed by checkpoint and
expert layout, with the most frequent entries most-recent in the initial LRU.
It can populate existing slots without increasing their memory allocation.
Both TP ranks must use identical expert IDs but load their own packed records.
Loading must finish before graph/kernel consumers read those slots.
Startup kernel warmups and any cache reset must not silently discard the seed.

`tools/check_cache_policies.py` replays historical rank-0 routing. Training uses
`io-schedule-varied-warmup-routes-rank0.jsonl`; evaluation uses
`cache-layout-heldout-warmup-routes-rank0.jsonl`. It reproduces the current
resident-first grouping, last-64-token frequency ordering and requested-set
protection when evicting. These are SHORT prompts (maximum 34 tokens in the
evaluation trace), not a current 4096-scheduler PP benchmark. Tokens <=16 are a
phase proxy; short PP tails can be included in TG. Draft experts are excluded
because their 128 slots cover the full draft expert set.

| Seed per layer (capacity stays 224) | TG demand misses | PP demand misses | Preload GiB/rank | Total reads incl. preload |
|---|---:|---:|---:|---:|
| empty mapping | 10,336 | 4,159 | 0 | 14,495 |
| 64 | 9,238 | 3,274 | 22.41 | 15,072 |
| 128 | 7,981 | 2,600 | 44.82 | 15,701 |
| up to 224 | 6,987 | 1,982 | 78.37 | 17,921 |

Full seeding reduces TG demand misses by 32.4%, but total reads including startup
increase by 23.6% over this finite trace. This moves I/O before requests; it is
NOT a measured TG speed gain. Startup cost, first-call TTFT, warmup overwrites,
long PP pollution and repeated-session amortization require an actual A/B.
The running model already has a warm cache, so the empty-map baseline must not
be presented as its current measured state.

## 2. PP transient slots versus TG hot slots

Our PP already consumes resident experts first and loads experts frequent near
the prompt tail later. It still lets PP misses enter the ordinary LRU. Upstream
uses a transient region to avoid evicting the decode hot set during PP.

A fixed split of our existing 224 slots is not automatically beneficial:

| Hot + transient slots/layer | TG demand misses | PP demand misses | PP execution groups |
|---|---:|---:|---:|
| existing 224 unified, empty initial map | 10,336 | 4,159 | 320 |
| seeded 192 + 32 | 10,563 | 5,606 | 660 |
| seeded 160 + 64 | 17,488 | 7,338 | 640 |

These candidates keep PP misses in fenced, sequentially reused transient slots
and do not promote them into the hot region. They are illustrative fixed-budget
policies, not an exhaustive search. Smaller hot regions hurt TG and splitting
PP increases kernel groups. With short prompts the ordinary cache is already
large enough for each call, so a PP-only path is especially unattractive.

A better candidate is conditional protection only for large PP routing sets,
returning the entire 224 slots to TG. An extra transient pool would preserve all
224 hot slots but needs new memory: 32 slots for every target layer cost
11.21 GiB/rank; sharing one 32-slot pool across layers would cost about 0.28 GiB,
but requires binding/remapping and CUDA lifetime changes. The current worker
startup headroom with ASR is already close to the 115 GiB guard. Do not silently
add memory, bypass that guard or reduce the 224-slot target cache.

## 3. Decoder last-128-token replay

Upstream `engine/v41_engine.py::_decode_loop` runs layers 0..20 over the whole
prompt, then `engine/model.py::decoder_replay` runs layers 21..39 over its last
128 tokens with `win_lo=S`. It carries encoder hidden states, HC pre-mix,
indexer top-k/candidates and produces the DSpark hidden taps before seeding the
draft. Thus it is more complete than simply omitting decoder work.

It nevertheless truncates each decoder's SWA history at the replay boundary.
Earlier rows in those final 128 depend on preceding rows; later decoder layers
propagate those differences. Upstream `verify_replay` explicitly calls prompts
longer than the window an approximation. DSpark verification does not restore
the omitted target context: it verifies against the same approximate target.
Original weight precision does not make the calculation equivalent.

Porting this to our pinned vLLM also requires correct position/compressor state,
SWA writes, DSpark seed/taps and prefix-cache validity. It must not publish
128-token prefix checkpoints as though all omitted decoder rows were evaluated.
Exact layer-dependent backward dependency ranges are a different, more involved
implementation than a constant 128-token tail in every decoder layer.

See [the earlier exact-state experiment](DECODER_ROWS_EXPERIMENT.md): final-layer
routed-expert-only selection preserved checked states but gave about +2% PP and
-7.96% matched-output TG, and remains OFF. That result neither validates nor
measures upstream's broader approximate replay.

## Decision

Frequency seeding is the first candidate for a controlled implementation and
real timing. Adaptive PP cache protection merits a long-prompt trace before
implementation; the fixed partition tested here is not justified. Constant
128-token decoder replay is possible only as an explicitly approximate mode,
not a drop-in optimization under the current computation-preservation condition.

Replay artifact: `results/cache-policy-review.json` (input SHA256 and limitations).
Rerun:

```sh
python3 tools/check_cache_policies.py \
  --train results/io-schedule-varied-warmup-routes-rank0.jsonl \
  --test results/cache-layout-heldout-warmup-routes-rank0.jsonl \
  --output results/cache-policy-review.json
```
