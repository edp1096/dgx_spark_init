"""Load-time GLM-5.3 o_proj transplant for the Entrpi vLLM image.

Adapted from MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (MIT).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:
    logger = logging.getLogger("glm53_ablit")

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$")
MTP_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mtp_block\.self_attn\.o_proj$")


class AblitError(RuntimeError):
    pass


def enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def tp_info() -> tuple[int, int]:
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
    except Exception:
        return 1, 0


def parse_layers(value: str) -> set[int]:
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lower, upper = (int(part) for part in item.split("-", 1))
            if upper < lower:
                raise AblitError(f"invalid ABLIT_LAYERS range: {item}")
            result.update(range(lower, upper + 1))
        else:
            result.add(int(item))
    if not result:
        raise AblitError("ABLIT_LAYERS is empty")
    return result


def unwrap(model: Any) -> Any:
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return model
    return getattr(language_model, "model", language_model)


def candidates(model: Any):
    seen: set[int] = set()
    for name, module in unwrap(model).named_modules():
        if id(module) in seen:
            continue
        match = LAYER_RE.search(name) or MTP_RE.search(name)
        if match:
            seen.add(id(module))
            yield name, int(match.group(1)), module, MTP_RE.search(name) is not None


def maybe_apply(model: Any) -> dict[str, Any] | None:
    if not enabled("ABLIT", False):
        return None

    directory = Path(os.environ.get("ABLIT_DIR", "/opt/glm53/ablit"))
    manifest_path = directory / "MANIFEST.json"
    if not manifest_path.is_file():
        raise AblitError(f"ABLIT=1 but {manifest_path} is missing; run manage.sh setup")
    manifest = json.loads(manifest_path.read_text())
    layers = parse_layers(os.environ.get("ABLIT_LAYERS", "0-44"))
    include_mtp = enabled("ABLIT_INCLUDE_MTP", False)
    metadata = {int(key): value for key, value in manifest.get("layers", {}).items()}
    world, rank = tp_info()
    edited: list[int] = []

    modules = list(candidates(model))
    if not include_mtp and modules and all(is_mtp for _, _, _, is_mtp in modules):
        logger.info("ablit: keeping original MTP weights")
        return {"edited_layers": [], "tp_world": world, "tp_rank": rank}

    for name, layer, module, is_mtp in modules:
        if layer not in layers or (is_mtp and not include_mtp):
            continue
        info = metadata.get(layer)
        if not info:
            raise AblitError(f"donor manifest has no layer {layer}")
        path = directory / f"L{layer}.bin"
        if not path.is_file() or path.stat().st_size != int(info["nbytes"]):
            raise AblitError(f"donor tensor is missing or incomplete: {path}")
        if info.get("dtype") != "BF16":
            raise AblitError(f"unsupported donor dtype for L{layer}: {info.get('dtype')}")
        shape = tuple(int(value) for value in info["shape"])
        donor = torch.from_file(
            str(path), shared=False, size=int(info["nbytes"]) // 2, dtype=torch.bfloat16
        ).reshape(shape)
        weight = getattr(module, "weight", None)
        if weight is None or not torch.is_tensor(weight) or weight.ndim != 2:
            raise AblitError(f"{name} has no two-dimensional weight")
        local_in = weight.shape[1]
        if shape[0] != weight.shape[0] or shape[1] != local_in * world:
            raise AblitError(
                f"{name} local shape {tuple(weight.shape)} does not match donor "
                f"{shape} at TP={world}"
            )
        shard = donor[:, rank * local_in : (rank + 1) * local_in]
        with torch.no_grad():
            weight.copy_(shard.to(device=weight.device, dtype=weight.dtype))
        edited.append(layer)
        logger.info("ablit: transplanted %s from donor layer %d", name, layer)

    if not edited:
        raise AblitError("ABLIT=1 but this model load matched no configured o_proj")
    if not include_mtp and set(edited) != layers:
        raise AblitError(f"incomplete donor application: expected {sorted(layers)}, got {sorted(edited)}")
    logger.info(
        "ablit: ON donor=%s revision=%s layers=%s TP=%d rank=%d",
        manifest.get("donor"),
        manifest.get("revision"),
        sorted(edited),
        world,
        rank,
    )
    return {"edited_layers": sorted(edited), "tp_world": world, "tp_rank": rank}
