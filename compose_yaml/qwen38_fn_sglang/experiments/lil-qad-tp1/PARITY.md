# LIL vLLM / SGLang TP1 comparison

This is a port of checkpoint support and selected kernels, not a claim that
two different serving engines become identical by copying command-line flags.

| Area | SGLang v3 | Reference vLLM |
| --- | --- | --- |
| Checkpoint | LIL revision `7c4f1bc1...`, unchanged bytes | Same revision |
| Main routed experts | B12X NVFP4 W4A4, checkpoint activation scales | B12X NVFP4 |
| MTP experts | B12X W4A16, BF16 activations | B12X W4A16 |
| Dense MXFP8 | B12X, dynamic activation quantization | B12X |
| Vision NVFP4 FC2 | B12X W4A16 | Native kernel fallback (Marlin) |
| QSA and MTP KV | FP8 E4M3, proper write/read descales | FP8 |
| QSA arithmetic | Existing SM121 KDA with descaled BF16 scratch | Engine-specific sparse attention |
| GDN | B12X 9043 prefill/decode/verify, native SGLang convolution and accept/track commit | B12X 9043 |
| PLE | Original NVFP4; checkpoint UVA for graph decode, bounded io_uring for eager prefill | NVFP4 disk io_uring outside attention graphs |
| Prefix / multiturn | Explicit cold/cached branch and two-turn tests | Engine-native prefix caching |
| 1M RoPE | YaRN factor 4, original 262144 | Same factor and original length |
| 1M prefill chunk | 4096 | 4096 |
| 1M request concurrency | 1 | 1 |
| KV pool | Explicit 1,048,576 tokens | Automatic 1,318,640 tokens in reference run |
| MTP | Three draft steps, four target verify positions; ko64k draft head | Four speculative tokens; reference vocabulary policy |
| Main LM head | Original BF16 | Runtime MXFP8 weight quantization with BF16 activations |
| MTP LM head | BF16 ko64k draft vocabulary | Runtime NVFP4 weight quantization with BF16 activations |
| Weight loading | Native SGLang bounded loader, indexed MTP shard filtering, no extra PLE file copy | B12X O_DIRECT managed-storage loader |
| Graphs / compilation | Native SGLang decode/verify CUDA graphs | vLLM AOT / full-and-piecewise graphs |

## Flags that must not be copied literally

- `fuse_act_quant=true` enables a vLLM compiler pattern pass. In pinned vLLM
  `b40673cd...`, it matches static FP8, dynamic FP8 groups 64/128 and NVFP4;
  it does **not** register a dense MXFP8 group-32 pattern. B12X fused experts
  already own their activation/quantization path. The flag is not evidence
  that all MXFP8 projections were fused, nor a missing generic SGLang option.
- `gpu-memory-utilization` and `mem-fraction-static` have different budgeting
  semantics. Token capacity and measured host availability must be compared,
  not the numeric flag alone.
- vLLM's model loader intercepts vLLM allocation and weight-transfer APIs.
  Importing that loader into SGLang would not correctly intercept SGLang's
  constructors and `.copy_` loaders. It is not enabled under a misleading name.
- `VLLM_USE_AOT_COMPILE`, its V2 runner and compiler pass configuration are
  engine internals. SGLang's own graph paths require their own qualification.

## Interpretation of measurements

The serving workload uses identical prompt bytes, no thinking, temperature
zero and 256 generated tokens. Both cold-prefix and repeated-prefix runs are
recorded. Engine-specific MTP, LM head and cache-pool differences above remain
visible; these are practical configured-runtime comparisons, not controlled
single-kernel benchmarks. Startup includes loading/compilation and must be
reported separately from prefill and generation. See `VALIDATION.md` for
completed checks and retained results. A configured context is never treated
as a completed input test.

## Matched serving workload results

Single measurements on gx10-2; 256 output tokens, temperature zero, thinking
disabled. Integer sequence correctness was checked for both engines.

| Input tokens | Prefix | SGLang total (s) | vLLM total (s) |
| ---: | --- | ---: | ---: |
| 1,082 | cold | 7.303 | 4.486 |
| 1,082 | reused | 4.893 | 4.584 |
| 16,442 | cold | 13.931 | 11.181 |
| 16,442 | reused | 4.978 | 6.069 |
| 64,058 | cold | 40.560 | 34.050 |
| 64,058 | reused | 5.050 | 5.963 |

The 1K cold SGLang case includes any first-use shape preparation left after
the 122-token warmup. These are application latency measurements, not
steady-state kernel microbenchmarks. Neither engine wins every case.
Retained values and first-content timings: `results/serving-comparison.json`;
raw vLLM records: `../lil-vllm-tp1/results/serving-benchmark-20260919/`.
