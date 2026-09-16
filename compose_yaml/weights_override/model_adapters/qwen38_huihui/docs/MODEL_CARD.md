---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE
library_name: transformers
pipeline_tag: image-text-to-text
base_model:
- RadixArk/Qwen3.8-Flash-Next-NVFP4
- huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF
tags:
- qwen3.8
- nvfp4
- modelopt
- sglang
- abliterated
- gguf-delta-transfer
---

# Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4

Community model combining [RadixArk NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) with changes extracted from [Huihui’s abliterated GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF). Distributed as safetensors, with RadixArk’s original MTP, vision weights and tokenizer retained.

## Tested

- **TP1 / one DGX Spark:** 64K context setting; basic text, code, tool-call and image checks passed.
- **TP2 / two DGX Sparks:** 1M context setting with runtime YaRN ×4; basic checks and a **66K-token retrieval request** passed. Full 1M input quality was not tested.

Changes were recovered from quantized GGUF weights, so this is not an exact Huihui BF16 reconstruction. Original activation scales were retained without recalibration. Comprehensive quality and refusal-removal benchmarks remain untested.

[Source revisions and weight hashes](PROVENANCE.json)

## Sources and license

[Qwen Community License 1.0](LICENSE). Credits: [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF), [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer), and [llama.cpp](https://github.com/ggml-org/llama.cpp).
