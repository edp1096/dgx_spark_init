---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/nemotron-3.5-asr-streaming-0.6b
pipeline_tag: automatic-speech-recognition
tags:
- gguf
- quantized
- q5_k
- nemo-speech
---

# Nemotron 3.5 ASR Streaming 0.6B — Q5_K GGUF

Q5_K conversion from [NVIDIA's original model](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), pinned [`ea30d66debe3740a08b573244286791d423d6b3e`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/ea30d66debe3740a08b573244286791d423d6b3e).

- **File:** `nemotron-3.5-asr-streaming-0.6b.q5_k.gguf`
- **Size:** 528,061,920 bytes (503.6 MiB).
- **Conversion:** quantized from an F16 conversion of the original NeMo checkpoint. The official Q8_0 GGUF supplies metadata, tensor ordering and protected tensors; Q5_K weights are derived from F16. Matrices whose row widths do not fit 256-element K-quant blocks stay F16; other protected tensors retain their original types. This is a mixed-precision Q5_K file, not a Q5_K_M preset.
- **Runtime:** Use [NeMo-Speech.cpp v0.2.0](https://github.com/NVIDIA/NeMo-Speech.cpp/releases/tag/v0.2.0)

```bash
nemo-speech transcribe audio.wav \
  --model nemotron-3.5-asr-streaming-0.6b.q5_k.gguf \
  --language ja-JP
```

