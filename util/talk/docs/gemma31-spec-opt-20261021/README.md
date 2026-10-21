# Gemma 4 31B speculative decoding

One DGX Spark, TP1, Huihui NVFP4 target. EXL3 bit labels describe assistant body weights; all use the same 6-bit head. TG is the median of three 512-token samples; PP uses the 9K prompt. Values are tokens/s.

| Configuration | Korean TG | Code TG | PP |
| --- | ---: | ---: | ---: |
| EXL3 2bit / MTP9 | 30.23 | 31.71 | 862.5 |
| EXL3 3bit / MTP7 | 29.42 | 37.10 | 859.7 |
| EXL3 4bit / MTP9 | 30.70 | 39.17 | 861.6 |
| DFlash 15 | 17.73 | 23.37 | 852.1 |
| Paired control / MTP9 | 30.65 | 39.08 | 854.2 |
| EXL3 4bit / MTP9 + Split-KV | 32.12 | 40.98 | 855.4 |
| Google BF16 / MTP9 + Split-KV | 26.38 | 31.90 | 857.2 |
| DFlash + Split-KV | 17.97 | 22.22 | 852.6 |

Split-KV TG: 32K 14.87 → 20.63; 64K 8.52 → 18.77.

Basic checks: control-b4-9: 12/12, splitkv-b4-9: 12/12. This is not a general quality benchmark.

32K/64K PP/TG, latency and acceptance: [results](summary.json). Reproduction settings and limits: [protocol](protocol.json). Raw outputs and commands: [raw](raw). Kernel changes: [patch](runtime.patch).

Source: [Google assistant](https://huggingface.co/google/gemma-4-31B-it-assistant), [DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash), [SGLang reference](https://github.com/sgl-project/sglang/pull/44830).
