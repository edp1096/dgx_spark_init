"""Persistent, authenticated Krea 2 checkpoint downloads from Civitai."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx


CACHE_ROOT = Path("/root/.cache/huggingface/media-models")
TOKEN_FILE = Path("/root/.cache/huggingface/media-secrets/civitai_token")
COMFY_DIFFUSION_ROOT = Path("/opt/ComfyUI/models/diffusion_models")
COMFY_TEXT_ENCODER_ROOT = Path("/opt/ComfyUI/models/text_encoders")


@dataclass(frozen=True)
class CivitaiCheckpoint:
    key: str
    label: str
    version_id: int
    file_id: int
    precision: str
    filename: str
    size: int
    sha256: str
    model_id: int = 2574319
    provider: str = "civitai"
    repo_id: str = ""
    remote_file: str = ""

    @property
    def source(self) -> str:
        if self.provider == "huggingface":
            return f"https://huggingface.co/{self.repo_id}"
        return f"https://civitai.com/models/{self.model_id}?modelVersionId={self.version_id}"

    @property
    def download_url(self) -> str:
        if self.provider == "huggingface":
            return f"https://huggingface.co/{self.repo_id}/resolve/main/{self.remote_file}"
        return f"https://civitai.com/api/download/models/{self.version_id}?fileId={self.file_id}"


CHECKPOINTS = {
    item.key: item
    for item in (
        CivitaiCheckpoint(
            "ray-v1", "Ray Artshoot V1 · FP8", 3104536, 2984527, "fp8",
            "rayArtshoot_krea2NSFWV1_fp8.safetensors", 13_141_773_512,
            "b50c464c99f99a6b78778e96a98803b33d842560c9b4f4a49b747caf266d1ad2",
        ),
        CivitaiCheckpoint(
            "ray-v2", "Ray Artshoot V2 · FP8", 3163740, 3044242, "fp8",
            "rayArtshoot_krea2NSFWV2_fp8.safetensors", 13_141_774_776,
            "9b68eda36b6a93d95421026d2fffa14d93705744bde51a4d7e25c383b0d777ab",
        ),
        CivitaiCheckpoint(
            "ray-v3", "Ray Artshoot V3 · INT8", 3228659, 3111055, "int8",
            "rayArtshoot_krea2NSFWV3_int8.safetensors", 13_149_148_024,
            "a3ba4dc1a16c6d83004f1d92c651a60237b7bc266e39416914108c3dccbe924b",
        ),
        CivitaiCheckpoint(
            "ray-v4", "Ray Artshoot V4 · INT8", 3249241, 3132783, "int8",
            "rayArtshoot_krea2NSFWV4_int8.safetensors", 12_821_825_036,
            "5af76f002c0f482a07d99f5a9b21f1eacca55f81313534299b11fc6eccba0187",
        ),
        CivitaiCheckpoint(
            "moody-v7", "Moody Krea 2 Mix V7.0 · NVFP4", 3209007, 3090679, "nvfp4",
            "moodyKrea2Mix_v70.safetensors", 8_807_648_196,
            "a06425056c2a5fee5267eb58a23d75b00ffaa2df1a6d8cdd328ff45473ee14ab",
            model_id=2731187,
        ),
        CivitaiCheckpoint(
            "moody-cutie-v4", "Moody Cutie Mix V4.0 · NVFP4", 3211049, 3092808, "nvfp4",
            "moodyCutieMixKrea2_v40.safetensors", 8_807_683_316,
            "ad11a6e8f5e2619d06a17ad35023fe3accffe0d24cc6ff94ea81658f50ff3aa4",
            model_id=2764429,
        ),
        CivitaiCheckpoint(
            "moody-amateur-v1", "Moody Amateur Mix V1.0 · NVFP4", 3230531, 3112927, "nvfp4",
            "moodyAmateurMixKrea2_v10.safetensors", 8_807_683_316,
            "1ea493bce696001f804528d310f00db7db35ed8d5232166b6c2da87d40332e4d",
            model_id=2859971,
        ),
        CivitaiCheckpoint(
            "chriscole-edit-v1.1", "Krea 2 Turbo Edit v1.1 · FP8", 0, 0, "fp8",
            "Krea2_turbo_uncensored_edit_v1.1-fp8_scaled.safetensors", 13_141_731_040,
            "5d803c7fe509a0840624cd1aeffd8d54d1a72810c90370efe5eff293ff0e0fdb",
            provider="huggingface",
            repo_id="ChrisColeTech/krea2-turbo-uncensored-v1.1-FP8",
            remote_file="split/diffusion_models/Krea2_turbo_uncensored_edit_v1.1-fp8_scaled.safetensors",
        ),
    )
}

HERETIC_TEXT_ENCODER = CivitaiCheckpoint(
    "identity-heretic", "Qwen3-VL 4B Heretic · INT8 ConvRot", 3099765, 2979509, "int8",
    "qwen3VLInstruct4bHeretic_int8Convrot.safetensors", 5_246_908_680,
    "e2ec7cf7f166ee6927144787a153ef02ea41f1f56c62878af68d7b5db59eae36",
    model_id=2728378,
)

# Civitai also provides full BF16 sources for V2 and V4.  They are retained as
# conversion inputs and are not linked into ComfyUI's selectable model folder.
BF16_SOURCES = {
    item.key: item
    for item in (
        CivitaiCheckpoint(
            "ray-v2", "Ray Artshoot V2 · BF16 source", 3163740, 3122609, "bf16",
            "rayArtshoot_krea2NSFWV2_bf16_source.safetensors", 26_280_498_488,
            "92af82f040b39eaabd2dcf7ed84a94edae915b66bea2859c3e66cb6dac160302",
        ),
        CivitaiCheckpoint(
            "ray-v4", "Ray Artshoot V4 · BF16 source", 3249241, 3132383, "bf16",
            "rayArtshoot_krea2NSFWV4_bf16_source.safetensors", 26_280_500_256,
            "0d26279c43f4570d58b3beab4ab848136b64cc926ea322af8d462e3ac8a7caaf",
        ),
    )
}

NVFP4_FILENAMES = {
    "ray-v2": "rayArtshoot_krea2NSFWV2_nvfp4.safetensors",
    "ray-v4": "rayArtshoot_krea2NSFWV4_nvfp4.safetensors",
}


def stored_token() -> str:
    try:
        return TOKEN_FILE.read_text().strip()
    except OSError:
        return ""


def save_token(token: str) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)


def token_configured() -> bool:
    return bool(stored_token() or os.environ.get("CIVITAI_API_KEY", "").strip())


def model_path(checkpoint: CivitaiCheckpoint) -> Path:
    return CACHE_ROOT / checkpoint.filename


def nvfp4_path(key: str) -> Path:
    return CACHE_ROOT / NVFP4_FILENAMES[key]


def nvfp4_ready(key: str) -> bool:
    path = nvfp4_path(key)
    return path.is_file() and path.stat().st_size > 1_000_000_000


def nvfp4_validation_path(key: str) -> Path:
    return nvfp4_path(key).with_suffix(".validated.json")


def nvfp4_validated(key: str) -> bool:
    return nvfp4_ready(key) and nvfp4_validation_path(key).is_file()


def ready(checkpoint: CivitaiCheckpoint) -> bool:
    path = model_path(checkpoint)
    return path.is_file() and path.stat().st_size == checkpoint.size


def link_model(checkpoint: CivitaiCheckpoint) -> None:
    source = model_path(checkpoint)
    if not ready(checkpoint):
        raise RuntimeError(f"checkpoint is incomplete: {checkpoint.key}")
    destination = COMFY_DIFFUSION_ROOT / checkpoint.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        if destination.resolve() == source.resolve():
            return
        raise RuntimeError(f"refusing to replace non-symlink model path: {destination}")
    destination.symlink_to(source)


def link_text_encoder(checkpoint: CivitaiCheckpoint = HERETIC_TEXT_ENCODER) -> None:
    source = model_path(checkpoint)
    if not ready(checkpoint):
        raise RuntimeError(f"text encoder is incomplete: {checkpoint.key}")
    destination = COMFY_TEXT_ENCODER_ROOT / checkpoint.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        if destination.resolve() == source.resolve():
            return
        raise RuntimeError(f"refusing to replace non-symlink model path: {destination}")
    destination.symlink_to(source)


def link_nvfp4(key: str) -> None:
    source = nvfp4_path(key)
    if not nvfp4_ready(key):
        raise RuntimeError(f"converted checkpoint is incomplete: {key}")
    destination = COMFY_DIFFUSION_ROOT / NVFP4_FILENAMES[key]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        if destination.resolve() == source.resolve():
            return
        raise RuntimeError(f"refusing to replace non-symlink model path: {destination}")
    destination.symlink_to(source)


def download_checkpoint(
    checkpoint: CivitaiCheckpoint,
    token: str,
    progress: Callable[[int, int], None] | None = None,
    link: bool = True,
) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    final = model_path(checkpoint)
    partial = final.with_suffix(final.suffix + ".part")
    if ready(checkpoint):
        if link:
            link_model(checkpoint)
        if progress:
            progress(checkpoint.size, checkpoint.size)
        return

    last_error: Exception | None = None
    for attempt in range(8):
        existing = partial.stat().st_size if partial.exists() else 0
        if existing > checkpoint.size:
            partial.unlink()
            existing = 0
        if existing == checkpoint.size:
            break
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            timeout = httpx.Timeout(connect=30.0, read=60.0, write=60.0, pool=30.0)
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                with client.stream("GET", checkpoint.download_url, headers=headers) as response:
                    if response.status_code not in {200, 206}:
                        raise RuntimeError(f"Civitai download failed with HTTP {response.status_code}")
                    append = existing > 0 and response.status_code == 206
                    if not append:
                        existing = 0
                    mode = "ab" if append else "wb"
                    written = existing
                    with partial.open(mode) as output:
                        for chunk in response.iter_bytes(8 * 1024 * 1024):
                            output.write(chunk)
                            written += len(chunk)
                            if progress:
                                progress(written, checkpoint.size)
            last_error = None
        except httpx.HTTPError as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 15))
            continue
        if partial.stat().st_size == checkpoint.size:
            break
    if last_error is not None and (not partial.exists() or partial.stat().st_size != checkpoint.size):
        raise RuntimeError(f"Civitai download failed after retries: {last_error}") from last_error
    if partial.stat().st_size != checkpoint.size:
        raise RuntimeError(
            f"Civitai download size mismatch for {checkpoint.key}: "
            f"{partial.stat().st_size} != {checkpoint.size}"
        )
    digest = hashlib.sha256()
    with partial.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != checkpoint.sha256:
        raise RuntimeError(f"Civitai checksum mismatch for {checkpoint.key}")
    partial.replace(final)
    if link:
        link_model(checkpoint)
