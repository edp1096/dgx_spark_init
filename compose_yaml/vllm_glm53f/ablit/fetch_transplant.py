#!/usr/bin/env python3
"""Fetch only GLM-5.3 Flash o_proj donor tensors for load-time transplant.

Derived from MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (MIT).
The donor checkpoint remains subject to its own model terms.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

DONOR = os.environ.get(
    "ABLIT_DONOR", "dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4"
)
REVISION = os.environ.get(
    "ABLIT_DONOR_REVISION", "f75389c0919bd6d3abd4f08c7a0f22545edc95d7"
)
LAYERS = range(15, 46)
CHUNK = 8 * 1024 * 1024
RETRIES = 4


def token() -> str:
    value = os.environ.get("HF_TOKEN", "").strip()
    if value:
        return value
    path = Path.home() / ".cache/huggingface/token"
    return path.read_text().strip() if path.is_file() else ""


def headers(start: int | None = None, end: int | None = None) -> dict[str, str]:
    result = {"User-Agent": "vllm-glm53f-ablit-fetch/1"}
    auth = token()
    if auth:
        result["Authorization"] = f"Bearer {auth}"
    if start is not None:
        result["Range"] = f"bytes={start}-{'' if end is None else end}"
    return result


def request(url: str, start: int | None = None, end: int | None = None):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers(start, end)), timeout=120
    )


def fetch_range(url: str, start: int, end: int, label: str) -> bytes:
    expected = end - start + 1
    for attempt in range(1, RETRIES + 1):
        try:
            with request(url, start, end) as response:
                data = bytearray()
                while len(data) < expected:
                    block = response.read(min(CHUNK, expected - len(data)))
                    if not block:
                        break
                    data.extend(block)
            if len(data) == expected:
                return bytes(data)
            raise RuntimeError(f"short read {len(data)}/{expected}")
        except Exception as exc:  # noqa: BLE001
            if attempt == RETRIES:
                raise SystemExit(f"failed to fetch {label}: {exc}") from exc
            print(f"retry {attempt}/{RETRIES} for {label}: {exc}", flush=True)
            time.sleep(3 * attempt)
    raise AssertionError("unreachable")


def json_url(base: str, name: str) -> dict:
    with request(f"{base}/{name}") as response:
        return json.loads(response.read())


def tensor_span(url: str, key: str) -> tuple[int, int, dict]:
    header_size = int.from_bytes(fetch_range(url, 0, 7, "safetensors size"), "little")
    raw_header = fetch_range(url, 8, 8 + header_size - 1, "safetensors header")
    header = json.loads(raw_header)
    if key not in header:
        raise SystemExit(f"{key} is absent from its indexed shard")
    metadata = header[key]
    begin, finish = metadata["data_offsets"]
    return 8 + header_size + begin, 8 + header_size + finish - 1, metadata


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def existing_ok(path: Path, metadata: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(metadata.get("nbytes", -1))
        and digest(path.read_bytes()) == metadata.get("sha256")
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{DONOR}/resolve/{REVISION}"
    index = json_url(base, "model.safetensors.index.json")["weight_map"]
    manifest_path = output / "MANIFEST.json"
    manifest = {
        "donor": DONOR,
        "revision": REVISION,
        "method": "dealign-oproj-transplant",
        "layers": {},
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("donor") == DONOR and previous.get("revision") == REVISION:
            manifest = previous

    for layer in LAYERS:
        key = f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
        shard = index.get(key)
        if not shard:
            raise SystemExit(f"donor index has no {key}")
        destination = output / f"L{layer}.bin"
        old = manifest.get("layers", {}).get(str(layer), {})
        if existing_ok(destination, old):
            print(f"L{layer}: already verified")
            continue
        url = f"{base}/{shard}"
        begin, end, metadata = tensor_span(url, key)
        if metadata.get("dtype") != "BF16":
            raise SystemExit(f"L{layer}: expected BF16, got {metadata.get('dtype')}")
        print(f"L{layer}: downloading {(end - begin + 1) / 1e6:.0f} MB from {shard}")
        data = fetch_range(url, begin, end, key)
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
        manifest.setdefault("layers", {})[str(layer)] = {
            "key": key,
            "shard": shard,
            "dtype": metadata["dtype"],
            "shape": metadata["shape"],
            "nbytes": len(data),
            "sha256": digest(data),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(int(item["nbytes"]) for item in manifest["layers"].values())
    print(f"Abliteration donor ready: {len(manifest['layers'])} tensors, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
