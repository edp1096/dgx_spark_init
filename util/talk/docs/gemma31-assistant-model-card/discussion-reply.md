The README's ExLlamaV3 requirement was incomplete. Stock ExLlamaV3 1.6.1 does not register `Gemma4AssistantForCausalLM`.

The backend we used was **SGLang Frozen-KV MTP with a custom EXL3 assistant loader**, available in this [archived recipe](https://github.com/edp1096/dgx_spark_init/tree/main/compose_yaml/zzz_not_use/gemma4_31b_sglang). SGLang runs the target and speculative verification; ExLlamaV3 1.5.4 CUDA matrices run the assistant's quantized linear operations. Unmodified SGLang is not sufficient for these EXL3 weights either.

This artifact is a draft model for the matching Gemma 4 31B target, not an independent chat model. It uses the target's embeddings, KV cache and intermediate state. The `ForCausalLM` class suffix does not remove those dependencies.

The tested recipe used a Huihui NVFP4 target on one DGX Spark. It is archived, and we do not claim stock ExLlamaV3 support, universal speedups or exact output equivalence for this custom quantized assistant.

Performance was modest: Korean whole-request throughput was 28.32 tok/s versus 27.38 with the original assistant; at 261,952 input tokens, cold time to first token was 39.1 minutes and decode was 6.91 tok/s. These are historical measurements, not a recommendation for fast long-context serving.
