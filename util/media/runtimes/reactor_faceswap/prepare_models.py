#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


MODELS = Path("/opt/ComfyUI/models")
INSIGHTFACE = MODELS / "insightface"
BUFFALO = INSIGHTFACE / "models" / "buffalo_l"
NSFW = MODELS / "nsfw_detector" / "vit-base-nsfw-detector"
INSWAPPER_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"


def copy_download(repo_id: str, filename: str, destination: Path, repo_type: str = "model") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return
    cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    temporary = destination.with_suffix(destination.suffix + ".part")
    shutil.copyfile(cached, temporary)
    temporary.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_insightface() -> None:
    swapper = INSIGHTFACE / "inswapper_128.onnx"
    copy_download("Gourieff/ReActor", "models/inswapper_128.onnx", swapper, "dataset")
    if sha256(swapper) != INSWAPPER_SHA256:
        swapper.unlink(missing_ok=True)
        raise RuntimeError("inswapper_128.onnx checksum mismatch")

    required = ("det_10g.onnx", "w600k_r50.onnx", "genderage.onnx")
    if all((BUFFALO / name).is_file() for name in required):
        return
    archive = INSIGHTFACE / "buffalo_l.zip"
    copy_download("Gourieff/ReActor", "models/buffalo_l.zip", archive, "dataset")
    BUFFALO.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(BUFFALO)
    archive.unlink(missing_ok=True)
    if not all((BUFFALO / name).is_file() for name in required):
        raise RuntimeError("buffalo_l archive is missing required face-analysis models")


def prepare_safety_model() -> None:
    for filename in ("config.json", "model.safetensors", "preprocessor_config.json"):
        copy_download("AdamCodd/vit-base-nsfw-detector", filename, NSFW / filename)


if __name__ == "__main__":
    prepare_insightface()
    prepare_safety_model()
    print("ReActor models are ready")
