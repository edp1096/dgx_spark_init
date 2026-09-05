"""Runtime rank-1 refusal-direction projection for Qwen3.8 Flash-Next."""
from __future__ import annotations

import os
import threading

import torch
from safetensors.torch import load_file

_path = os.environ.get("EXL3_ABLIT_DIRECTION", "").strip()
_lambda = float(os.environ.get("EXL3_ABLIT_LAMBDA", "1.5"))
_cpu_direction = None
_device_directions: dict[tuple[str, torch.dtype], torch.Tensor] = {}
_lock = threading.Lock()

if _path:
    _cpu_direction = load_file(_path, device="cpu")["direction"].float().contiguous()
    if _cpu_direction.ndim != 1:
        raise ValueError("EXL3 ablation direction must be a rank-1 tensor")
    norm = torch.linalg.vector_norm(_cpu_direction).item()
    if not 0.999 <= norm <= 1.001:
        raise ValueError(f"EXL3 ablation direction is not normalized: {norm}")
    print(
        f" == runtime ablation enabled: lambda={_lambda:g}, hidden={_cpu_direction.numel()}",
        flush=True,
    )


def enabled() -> bool:
    return _cpu_direction is not None


def _direction_for(x: torch.Tensor) -> torch.Tensor:
    key = (str(x.device), x.dtype)
    direction = _device_directions.get(key)
    if direction is None:
        with _lock:
            direction = _device_directions.get(key)
            if direction is None:
                direction = _cpu_direction.to(device=x.device, dtype=x.dtype)
                _device_directions[key] = direction
    return direction


def project_writer(x: torch.Tensor) -> torch.Tensor:
    """Apply (I - lambda vv^T) to the last residual-stream dimension."""
    if _cpu_direction is None:
        return x
    if x.shape[-1] != _cpu_direction.numel():
        raise ValueError(
            f"EXL3 ablation hidden-size mismatch: output={x.shape[-1]}, "
            f"direction={_cpu_direction.numel()}"
        )
    direction = _direction_for(x)
    coefficient = torch.sum(x * direction, dim=-1, keepdim=True)
    return x - (_lambda * coefficient) * direction


def project_mtp_linear(key: str, x: torch.Tensor) -> torch.Tensor:
    if key in ("mtp.fc_hidden", "mtp.fc_embedding"):
        return project_writer(x)
    return x


def project_embedding(key: str, x: torch.Tensor) -> torch.Tensor:
    if key == "model.language_model.embed_tokens":
        return project_writer(x)
    return x


def project_block(key: str, x: torch.Tensor) -> torch.Tensor:
    if key.startswith("model.language_model.layers.") or key.startswith("mtp.layers."):
        return project_writer(x)
    return x
