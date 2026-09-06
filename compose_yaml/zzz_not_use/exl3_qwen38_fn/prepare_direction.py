#!/usr/bin/env python3
"""Recover the published Flash-Next refusal direction from one rank-1 writer delta."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

import requests
import torch
from safetensors.torch import save_file

BASE_REPO = os.environ.get("ABLIT_BASE_REPO", "Qwen/Qwen3.8-Flash-Next")
BASE_REVISION = os.environ.get(
    "ABLIT_BASE_REVISION", "de4b8e4d43b917e7706784d8bb445c9af86a3540"
)
REFERENCE_REPO = os.environ.get(
    "ABLIT_REFERENCE_REPO", "windowsxp811203/Qwen3.8-Flash-Next-Abliterated"
)
REFERENCE_REVISION = os.environ.get(
    "ABLIT_REFERENCE_REVISION", "deb02632504bb214702bc28b0381a93d3112f500"
)
TENSOR_NAME = os.environ.get(
    "ABLIT_DIRECTION_TENSOR",
    "model.language_model.layers.20.linear_attn.out_proj.weight",
)
OUTPUT_DIR = Path(os.environ.get("ABLIT_OUTPUT_DIR", "/ablit"))
LAMBDA = float(os.environ.get("EXL3_ABLIT_LAMBDA", "1.5"))
TOKEN = os.environ.get("HF_TOKEN", "").strip()
TIMEOUT = (30, 300)


def url(repo: str, revision: str, filename: str) -> str:
    return (
        f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/"
        f"{quote(filename, safe='/')}"
    )


def headers(byte_range: tuple[int, int] | None = None) -> dict[str, str]:
    result = {"Accept-Encoding": "identity"}
    if TOKEN:
        result["Authorization"] = f"Bearer {TOKEN}"
    if byte_range:
        result["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return result


def get_json(repo: str, revision: str, filename: str) -> dict:
    response = requests.get(url(repo, revision, filename), headers=headers(), timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_range(repo: str, revision: str, filename: str, start: int, end: int) -> bytes:
    response = requests.get(
        url(repo, revision, filename),
        headers=headers((start, end)),
        timeout=TIMEOUT,
        stream=True,
    )
    try:
        if response.status_code != 206:
            raise RuntimeError(
                f"range request was not honored for {repo}/{filename}: HTTP {response.status_code}"
            )
        data = response.content
    finally:
        response.close()
    expected = end - start + 1
    if len(data) != expected:
        raise RuntimeError(f"short range read for {filename}: {len(data)} != {expected}")
    return data


def load_remote_tensor(repo: str, revision: str, tensor_name: str) -> torch.Tensor:
    index = get_json(repo, revision, "model.safetensors.index.json")
    shard = index["weight_map"][tensor_name]
    header_len = struct.unpack("<Q", get_range(repo, revision, shard, 0, 7))[0]
    header = json.loads(get_range(repo, revision, shard, 8, 7 + header_len))
    meta = header[tensor_name]
    if meta["dtype"] != "BF16":
        raise RuntimeError(f"expected BF16 direction source, got {meta['dtype']}")
    begin, finish = meta["data_offsets"]
    raw = get_range(repo, revision, shard, 8 + header_len + begin, 8 + header_len + finish - 1)
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(meta["shape"])
    print(f"read {repo}@{revision[:12]}:{tensor_name} {tuple(tensor.shape)}", flush=True)
    return tensor


def existing_is_current(manifest_path: Path, direction_path: Path) -> bool:
    if not manifest_path.is_file() or not direction_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    return all(
        (
            manifest.get("base_revision") == BASE_REVISION,
            manifest.get("reference_revision") == REFERENCE_REVISION,
            manifest.get("tensor") == TENSOR_NAME,
            manifest.get("lambda") == LAMBDA,
        )
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    direction_path = OUTPUT_DIR / "direction.safetensors"
    manifest_path = OUTPUT_DIR / "direction.json"
    if existing_is_current(manifest_path, direction_path):
        print(f"direction already prepared: {direction_path}")
        return 0

    base = load_remote_tensor(BASE_REPO, BASE_REVISION, TENSOR_NAME).float()
    reference = load_remote_tensor(REFERENCE_REPO, REFERENCE_REVISION, TENSOR_NAME).float()
    if base.shape != reference.shape or base.ndim != 2:
        raise RuntimeError(f"unexpected writer shapes: {tuple(base.shape)} / {tuple(reference.shape)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    delta = (reference - base).to(device)
    u, singular, _ = torch.linalg.svd(delta, full_matrices=False)
    direction = u[:, 0]
    direction /= torch.linalg.vector_norm(direction)
    pivot = torch.argmax(torch.abs(direction))
    if direction[pivot] < 0:
        direction = -direction

    rank1_energy = (singular[0].square() / singular.square().sum()).item()
    base_device = base.to(device)
    reference_device = reference.to(device)
    base_leak = torch.linalg.vector_norm(direction @ base_device).item()
    reference_leak = torch.linalg.vector_norm(direction @ reference_device).item()
    leakage_ratio = reference_leak / base_leak
    relative_delta = (
        torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(base_device)
    ).item()
    if rank1_energy < 0.995:
        raise RuntimeError(f"reference delta is not rank-1 enough: energy={rank1_energy:.8f}")
    if not 0.45 <= leakage_ratio <= 0.55:
        raise RuntimeError(f"reference does not match lambda=1.5 projection: ratio={leakage_ratio:.6f}")

    direction = direction.float().cpu().contiguous()
    digest = hashlib.sha256(direction.numpy().tobytes()).hexdigest()
    manifest = {
        "method": "rank-1 left singular vector recovered from published BF16 writer delta",
        "base_repo": BASE_REPO,
        "base_revision": BASE_REVISION,
        "reference_repo": REFERENCE_REPO,
        "reference_revision": REFERENCE_REVISION,
        "tensor": TENSOR_NAME,
        "lambda": LAMBDA,
        "rank1_energy": rank1_energy,
        "leakage_ratio": leakage_ratio,
        "relative_delta": relative_delta,
        "direction_sha256": digest,
    }
    with tempfile.TemporaryDirectory(dir=OUTPUT_DIR) as tmpdir:
        tmp_direction = Path(tmpdir) / "direction.safetensors"
        tmp_manifest = Path(tmpdir) / "direction.json"
        save_file({"direction": direction}, tmp_direction)
        tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_direction, direction_path)
        os.replace(tmp_manifest, manifest_path)
    print(
        f"direction ready: {direction_path} "
        f"(rank1_energy={rank1_energy:.8f}, leakage_ratio={leakage_ratio:.6f})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"direction preparation failed: {exc}", file=sys.stderr)
        raise
