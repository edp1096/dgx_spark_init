"""Keep the NVIDIA card body intact beneath the derived-model introduction."""
def render(source, validation="Runtime validation pending."):
    if not source.startswith("---\n") or "\n---\n" not in source[4:]:
        raise ValueError("Expected NVIDIA model-card YAML metadata")
    metadata, body = source[4:].split("\n---\n", 1)
    original = "base_model:\n- zai-org/GLM-5.3-Flash"
    if original not in metadata:
        raise ValueError("Unexpected NVIDIA base_model metadata")
    metadata = metadata.replace(original, "base_model:\n- nvidia/GLM-5.3-Flash-NVFP4\n- huihui-ai/GLM-5.3-Flash-abliterated-GGUF", 1)
    return ("---\n" + metadata + "\n---\n\n"
            "# Huihui-GLM-5.3-Flash-abliterated-NVFP4\n\n"
            "Base:\n"
            "* [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)\n"
            "* [huihui-ai/GLM-5.3-Flash-abliterated-GGUF](https://huggingface.co/huihui-ai/GLM-5.3-Flash-abliterated-GGUF)\n\n"
            "Conversion: DQ(NVIDIA NVFP4) + DQ(Huihui GGUF) - DQ(Unsloth GGUF). "
            "Original activation scales are retained. GGUF residuals remain; equivalence to Huihui BF16 is not claimed. "
            "See `transfer-manifest.json`.\n\n" + validation + "\n\n"
            "NVIDIA's original model card follows. This derived model is not an official NVIDIA release.\n\n"
            "----\n" + body)
