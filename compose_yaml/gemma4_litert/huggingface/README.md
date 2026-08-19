---
library_name: litert-lm
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model:
- huihui-ai/Huihui-gemma-4-E2B-it-abliterated
tags:
- gemma-4
- litert-lm
- arm64
- int8
- multimodal
- abliterated
- uncensored
---

# Huihui Gemma 4 E2B Abliterated LiteRT-LM

Multimodal LiteRT-LM conversion of
[`huihui-ai/Huihui-gemma-4-E2B-it-abliterated`](https://huggingface.co/huihui-ai/Huihui-gemma-4-E2B-it-abliterated).

The bundle contains INT8 text prefill/decode, external and per-layer embedders,
a Gemma 4 vision encoder, and a vision adapter. It was converted natively on a
Linux ARM64 NVIDIA DGX Spark; no x86 emulation was used.

## Conversion settings

- LiteRT Torch task: `image_text_to_text`
- Quantization: `dynamic_wi8_afp32` for text and vision
- Prefill signatures: 128, 256, and 512 tokens
- KV cache: 4,096 tokens
- Image soft tokens: 280
- Audio encoder: not included

## Usage

```bash
litert-lm import \
  Huihui-gemma-4-E2B-it-abliterated.litertlm \
  huihui-gemma4-e2b

litert-lm run huihui-gemma4-e2b \
  --backend gpu \
  --vision-backend gpu \
  --attachment image.png \
  --prompt "Describe this image."
```

The resulting bundle is approximately 4.9 GiB. Text and image inference were
both validated through the LiteRT-LM CLI and its OpenAI-compatible API.

## Safety

The source is an abliterated model with substantially reduced refusal behavior.
It can produce sensitive, controversial, or inappropriate content. Review its
outputs and do not assume that it provides default safety guarantees.
