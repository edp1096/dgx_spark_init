# Guarded context-growth check

> Cleanup 2026-09-13: rejected prototype source, experiment-only launch paths and dedicated replay tools were removed. The descriptions and commands below are historical records, not currently available runtime features. Adopted optimizations, safety guards and measurement evidence are retained.

2026-09-13: preserve KV 2 GiB/rank, expert cache, native_hint and next-chunk
prefetch. Start at max_model_len 98,304 with stricter probe guards: host
MemAvailable >= 10 GiB and cgroup current <= 96 GiB, alongside the existing
100 GiB hard container budget. Both independently-running watchers confirm
these thresholds in their startup records.

The head crossed the host floor during initial FlashInfer MXFP8 autotuning:
9.967 GiB available, 89.271 GiB in the container. Its watcher killed only the
labelled probe container. OOM counters were zero; both hosts remained on the
same boot IDs/kernel. No long input was submitted. The 131,072 step was not
attempted, as promised when a guard trips. Existing 65,536 serving is restored
and verified separately.

This does **not** establish that 96K cannot fit or that the 2 GiB KV pool is
full. It identifies a startup peak under a deliberately conservative floor.
A future attempt should examine/reuse the startup autotune cache or otherwise
reduce that peak, rather than silently weakening the safety guard. No YaRN
factor or KV quantization was changed.

[Guard result](../results/context-growth/96k-summary.json).
