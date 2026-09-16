# TP2 tuning experiments

Baseline: local Huihui-RadixArk NVFP4, SGLang TP2, BF16 KV, 1M context,
NEXTN 3/1/4, full draft vocabulary, prefill chunk 1024.

1. Build balanced TP shards for the tokenizer-checked Korean 64K draft shortlist.
   `tp_vocab.py` reduces startup weights once; the normal TP logits gather remains.
   Target weights and full-vocabulary verification are unchanged.
2. Compare 1024/2048/4096/8192 prefill chunks with the same prompts and KV capacity.
3. Compare Korean output with current sampling, top_p=0.8, and a Korean-language
   instruction. Do not infer a language-quality improvement from speed alone.

`measure.py` records response text, timings, usage and speculative verification counts.
`memory.py` records both hosts; the existing labelled-container watchdog remains
active (8 GiB host floor, 98 GiB cgroup ceiling). `start_variant.py` keeps the
1M context and verifies startup settings. Runtime experiments live in `runtime/`.

Reference reviewed: tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark,
commit 6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0, reduced-vocabulary TP ownership.
Implementation here balances arbitrary Korean shortlist positions across ranks
rather than limiting candidates to the lowest consecutive token IDs.

Raw measurements and conclusions: `results/`. Preliminary sweeps that overlapped
image transfer are labelled and must not be used as clean A/B speed evidence.
