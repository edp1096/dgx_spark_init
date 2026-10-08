"""Convert a full-precision Krea 2 DiT checkpoint to ComfyUI NVFP4."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import safetensors
import safetensors.torch
import torch
from comfy_kitchen.tensor import TensorCoreNVFP4Layout


# These projections are unusually sensitive.  This conservative Krea 2 profile
# follows the public Kitchen converter and keeps the complete text-fusion path in
# BF16; the large image transformer blocks still receive native NVFP4 weights.
BF16_COMPONENTS = ("first", "last", "tmlp", "tproj", "txtfusion", "txtmlp")
PROFILE_SOURCE = "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
PROFILE_COMMIT = "2eabdc38abde1337a73f35fa90977322d3305965"


def _layer_name(weight_key: str) -> str:
    name = weight_key.removesuffix(".weight")
    return name.removeprefix("model.diffusion_model.")


def convert_krea2_nvfp4(
    source: Path,
    destination: Path,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Quantize atomically and return tensor/layer counts."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    state = safetensors.torch.load_file(source, device="cpu")
    output: dict[str, torch.Tensor] = {}
    layers: dict[str, dict[str, object]] = {}
    total = len(state)
    quantized = 0

    for index, (key, value) in enumerate(state.items(), start=1):
        if value.ndim == 2 and key.endswith(".weight") and not any(
            component in key for component in BF16_COMPONENTS
        ):
            weight = value.to(device="cuda", dtype=torch.bfloat16)
            qdata, params = TensorCoreNVFP4Layout.quantize(weight)
            for suffix, tensor in TensorCoreNVFP4Layout.state_dict_tensors(qdata, params).items():
                if tensor.dtype == torch.float8_e8m0fnu:
                    tensor = tensor.view(torch.uint8)
                output[f"{key}{suffix}"] = tensor.cpu()
            layers[_layer_name(key)] = {"format": "nvfp4"}
            del weight, qdata, params
            quantized += 1
        elif value.is_floating_point():
            output[key] = value.to(dtype=torch.bfloat16)
        else:
            output[key] = value
        if progress:
            progress(index, total)

    metadata = OrderedDict(
        _quantization_metadata=json.dumps({"format_version": "1.0", "layers": layers}),
        format="pt",
        converted_by="Spark Media Krea 2 NVFP4 converter",
        quantization_profile=PROFILE_SOURCE,
        quantization_profile_commit=PROFILE_COMMIT,
    )
    safetensors.torch.save_file(output, partial, metadata=metadata)
    del output, state
    with safetensors.safe_open(partial, framework="pt") as handle:
        keys = list(handle.keys())
        saved_metadata = handle.metadata() or {}
    if quantized < 100 or "_quantization_metadata" not in saved_metadata:
        partial.unlink(missing_ok=True)
        raise RuntimeError("NVFP4 validation failed: quantized layer metadata is incomplete")
    partial.replace(destination)
    torch.cuda.empty_cache()
    return {"source_tensors": total, "output_tensors": len(keys), "quantized_layers": quantized}
