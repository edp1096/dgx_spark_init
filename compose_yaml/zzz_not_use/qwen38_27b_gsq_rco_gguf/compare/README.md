# 27B deployment comparison

Run from this directory or any working directory:

```bash
python3 run.py --output ../data/comparison-20260908 \
  nvfp4-base nvfp4-dflash2 exl3-base exl3-mtp gguf-base gguf-mtp3 bf16-control
python3 analyze.py ../data/comparison-20260908
```

An additional measured memory variant limits llama.cpp's global prompt cache
from its default 8192 MiB to 1024 MiB, keeping the same 32K context and MTP depth:

```bash
python3 run.py --output ../data/comparison-20260908 gguf-mtp3-cache1g
python3 analyze.py ../data/comparison-20260908
```

The runner starts one isolated container at a time on localhost:18697, records its
exact launch arguments and logs, and removes that container afterward. All other
GPU servers must already be stopped. It does not start or stop existing services.
The model paths and image versions are deliberately pinned to this machine's
installed deployments; see `launch()` before reusing on another machine.

## Measurement scope

- Context capacity: 32,768 tokens; one client request at a time; CPU affinity
  5–9,15–19 for every runtime.
- SGLang's total KV token pool is also capped at 32,768 with
  `--max-total-tokens`; without this, its existing 0.38 memory-fraction preset
  allocated roughly 704,000 cache tokens despite a 32K per-request context.
- Four workloads from `validate.py`: Python coding, Bayesian probability,
  Korean technical explanation, and Korean prose; two nonstreaming repetitions.
- Output cap 512 tokens, temperature 0.6, top-p 0.95, top-k 20, seed 42,
  thinking disabled. An EOS may end the output earlier. Identical seed values
  do not imply identical random streams across engines.
- Common speed metric: sum of API-reported output tokens divided by sum of
  client elapsed seconds. This includes prefill and API overhead. Native llama.cpp
  decode timings are reported separately when available.
- Four additional streamed requests with an output cap of 128 measure time to
  first nonempty content/reasoning/tool delta. These follow the repeated workloads,
  so they measure warm, potentially cached response latency. EXL3's API buffers
  a small tail before emitting text; this is part of the observed API latency.
- Quality probes use greedy sampling. Retain every request/response and inspect
  factual prose manually. Code is executed against seven functional cases in a
  disposable container without network, host mounts, or GPU access.
- Long-input retrieval is called twice to expose prefix reuse. Its latency
  includes generation; it is not a pure prefill timing.
- Memory is host `MemAvailable` before startup, after readiness and after tests,
  appropriate to GB10 shared memory but affected by allocator reservations and
  other host activity. File sizes are stored separately. This is not a peak
  process-memory or per-weight memory measurement.

## Interpretation limits

These are three **deployed checkpoints and runtimes**, not a controlled comparison
of quantizers applied to identical weights. EXL3 uses a different uncensored base;
NVFP4 contains mixed precision; KV cache formats also differ. The GGUF is the
existing ISTA allocation/imatrix-based IQ3_S build, not a new GSQ or RCO training
run. BF16 is the source of that GGUF and is used only as a quality control.

This small test set cannot establish general model intelligence or a standardized
benchmark score. Tool parsing and image support are evaluated through each
installed API, so failures can be server capability limitations.
