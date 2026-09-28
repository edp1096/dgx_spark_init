#!/usr/bin/env python3
"""Requantize the uncensored FLUX.2 Qwen text encoder to ComfyUI NVFP4.

The official Comfy checkpoint is used only as a mixed-precision layout
template. Every weight value is read from the uncensored checkpoint and
requantized; official weight values are never copied into the output.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path

import torch
from comfy.quant_ops import TensorCoreFP8E4M3Layout, TensorCoreNVFP4Layout
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="uncensored FP16/BF16 safetensors")
    parser.add_argument("--template", help="official Comfy mixed FP4 safetensors")
    parser.add_argument("--output", help="new uncensored mixed NVFP4 safetensors")
    return parser.parse_args()


def quant_config(template, key: str) -> tuple[str | None, torch.Tensor | None]:
    if not key.endswith(".weight"):
        return None, None
    metadata_key = key.removesuffix(".weight") + ".comfy_quant"
    if metadata_key not in template.keys():
        return None, None
    raw = template.get_tensor(metadata_key)
    config = json.loads(raw.numpy().tobytes())
    return config.get("format"), raw


def cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().cpu()


def safetensors_dtype(dtype: torch.dtype) -> str:
    names = {
        torch.bfloat16: "BF16",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.uint8: "U8",
        torch.float32: "F32",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported output dtype: {dtype}") from exc


def main() -> None:
    args = arguments()
    source_path = Path(
        args.source
        or hf_hub_download(
            "ponpoke/flux2-klein-4b-uncensored-text-encoder",
            "flux2-klein-4b-uncensored-text-encoder/model.safetensors",
        )
    ).resolve()
    template_path = Path(
        args.template
        or hf_hub_download(
            "Comfy-Org/flux2-klein-4B",
            "split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors",
        )
    ).resolve()
    output_path = Path(
        args.output
        or Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
        / "local"
        / "flux2-klein-4b-uncensored-nvfp4"
        / "qwen_3_4b_uncensored_nvfp4_flux2.safetensors"
    ).resolve()
    if output_path in {source_path, template_path}:
        raise ValueError("output must not overwrite the source or template")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output: dict[str, torch.Tensor] = {}
    counts = {"bf16": 0, "fp8": 0, "nvfp4": 0}
    torch.cuda.reset_peak_memory_stats()

    with (
        safe_open(source_path, framework="pt", device="cpu") as source,
        safe_open(template_path, framework="pt", device="cpu") as template,
    ):
        source_keys = list(source.keys())
        template_keys = set(template.keys())
        for index, key in enumerate(source_keys, 1):
            tensor = source.get_tensor(key)
            quant_format, metadata = quant_config(template, key)
            prefix = key.removesuffix(".weight")

            if quant_format == "nvfp4":
                quantized, params = TensorCoreNVFP4Layout.quantize(tensor.cuda())
                output[key] = cpu(quantized)
                output[prefix + ".weight_scale"] = cpu(params.block_scale)
                output[prefix + ".weight_scale_2"] = cpu(params.scale)
                output[prefix + ".comfy_quant"] = metadata
                counts["nvfp4"] += 1
            elif quant_format == "float8_e4m3fn":
                quantized, params = TensorCoreFP8E4M3Layout.quantize(
                    tensor.cuda(), scale="recalculate"
                )
                output[key] = cpu(quantized)
                output[prefix + ".weight_scale"] = cpu(params.scale)
                output[prefix + ".comfy_quant"] = metadata
                counts["fp8"] += 1
            elif quant_format is None:
                output[key] = tensor.to(torch.bfloat16).contiguous()
                counts["bf16"] += 1
            else:
                raise ValueError(f"unsupported template format {quant_format!r} for {key}")

            del tensor
            if index % 20 == 0 or index == len(source_keys):
                print(
                    f"[{index:03d}/{len(source_keys)}] {key} "
                    f"bf16={counts['bf16']} fp8={counts['fp8']} nvfp4={counts['nvfp4']}",
                    flush=True,
                )

        missing = template_keys.difference(output)
        extra = set(output).difference(template_keys)
        if missing or extra:
            raise RuntimeError(
                f"output/template key mismatch: missing={sorted(missing)[:10]} "
                f"extra={sorted(extra)[:10]}"
            )
        for key, tensor in output.items():
            expected = template.get_slice(key)
            if list(tensor.shape) != expected.get_shape() or safetensors_dtype(tensor.dtype) != expected.get_dtype():
                raise RuntimeError(
                    f"output/template tensor mismatch for {key}: "
                    f"got {tuple(tensor.shape)} {tensor.dtype}, "
                    f"expected {expected.get_shape()} {expected.get_dtype()}"
                )

    fd, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        save_file(
            output,
            temporary_path,
            metadata={
                "format": "pt",
                "source": "uncensored text encoder, official mixed-precision layout",
            },
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    del output
    gc.collect()
    print(f"saved {output_path} ({output_path.stat().st_size / 1024**3:.2f} GiB)")
    print(f"layers {counts}")
    print(f"peak CUDA allocated {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
