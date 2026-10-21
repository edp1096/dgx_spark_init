---
license: apache-2.0
base_model: google/gemma-4-31B-it-assistant
base_model_relation: quantized
tags:
  - exl3
  - gemma4
  - speculative-decoding
  - mtp
---

# Gemma 4 31B IT Assistant EXL3

MTP assistant with 4-bit linear weights and a 6-bit output head.

Requires the Gemma 4 31B target and a [custom SGLang loader](https://github.com/edp1096/dgx_spark_init/tree/main/compose_yaml/zzz_not_use/gemma4_31b_sglang). Not a standalone LLM.

Source: [Google Gemma 4 31B IT Assistant](https://huggingface.co/google/gemma-4-31B-it-assistant). Apache 2.0.
