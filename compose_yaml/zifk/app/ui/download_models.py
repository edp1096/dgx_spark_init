"""Model download, FP8 conversion, and status checker for ZIFK.

Downloads Z-Image Turbo/Base (BF16 → in-place FP8 convert, delete BF16 shards)
and FLUX.2 Klein 4B (official FP8).
"""

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from zifk_config import (
    KLEIN_4B_FP8_REPO,
    KLEIN_AE_FILE,
    KLEIN_AE_REPO,
    KLEIN_BASE_4B_FP8_REPO,
    KLEIN_BASE_MODEL_FILE,
    KLEIN_MODEL_FILE,
    KLEIN_TEXT_ENCODER_REPO,
    MODEL_DIR,
    ZIMAGE_BASE_DIR,
    ZIMAGE_BASE_REPO,
    ZIMAGE_TURBO_DIR,
    ZIMAGE_TURBO_REPO,
)

FP8_TRANSFORMER_FILENAME = "model_fp8.safetensors"


# ---------------------------------------------------------------------------
# Z-Image downloads (BF16 → in-place FP8 conversion)
# ---------------------------------------------------------------------------
def download_zimage_turbo(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    dest = model_dir / ZIMAGE_TURBO_DIR
    fp8_file = dest / "transformer" / FP8_TRANSFORMER_FILENAME
    if fp8_file.exists():
        print(f"[OK] Z-Image-Turbo (FP8) already exists")
        return
    if not dest.exists() or not any(dest.rglob("*.safetensors")):
        print(f"[DL] Downloading Z-Image-Turbo -> {dest}")
        snapshot_download(
            ZIMAGE_TURBO_REPO,
            local_dir=str(dest),
            ignore_patterns=["*.md", ".gitattributes"],
        )
    convert_zimage_fp8(dest, "Z-Image-Turbo")


def download_zimage_base(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    dest = model_dir / ZIMAGE_BASE_DIR
    fp8_file = dest / "transformer" / FP8_TRANSFORMER_FILENAME
    if fp8_file.exists():
        print(f"[OK] Z-Image-Base (FP8) already exists")
        return
    if not dest.exists() or not any(dest.rglob("*.safetensors")):
        print(f"[DL] Downloading Z-Image-Base -> {dest}")
        snapshot_download(
            ZIMAGE_BASE_REPO,
            local_dir=str(dest),
            ignore_patterns=["*.md", ".gitattributes"],
        )
    convert_zimage_fp8(dest, "Z-Image-Base")


# ---------------------------------------------------------------------------
# Z-Image FP8 in-place conversion
# ---------------------------------------------------------------------------
def convert_zimage_fp8(model_path: Path, label: str):
    """Convert transformer BF16 → FP8 in-place, delete BF16 shards.

    Result: model_path/transformer/ contains only config.json + model_fp8.safetensors
    """
    import torch
    from safetensors.torch import load_file, save_file

    transformer_dir = model_path / "transformer"
    fp8_file = transformer_dir / FP8_TRANSFORMER_FILENAME

    if fp8_file.exists():
        print(f"[OK] {label} FP8 already converted")
        return

    if not transformer_dir.exists():
        print(f"[SKIP] {label} transformer dir not found")
        return

    print(f"[CVT] Converting {label} transformer to FP8 (in-place)...")

    # Load all BF16 transformer safetensors (may be sharded)
    state_dict = {}
    index_files = list(transformer_dir.glob("*.safetensors.index.json"))
    bf16_files = []

    if index_files:
        import json
        with open(index_files[0]) as f:
            index = json.load(f)
        shard_files = set(index.get("weight_map", {}).values())
        for shard in shard_files:
            shard_path = transformer_dir / shard
            shard_dict = load_file(str(shard_path), device="cpu")
            state_dict.update(shard_dict)
            bf16_files.append(shard_path)
        bf16_files.extend(index_files)
    else:
        for sf in transformer_dir.glob("*.safetensors"):
            if sf.name == FP8_TRANSFORMER_FILENAME:
                continue
            shard_dict = load_file(str(sf), device="cpu")
            state_dict.update(shard_dict)
            bf16_files.append(sf)

    if not state_dict:
        print(f"[ERR] No safetensors files found in {transformer_dir}")
        return

    # Cast to FP8 e4m3fn (2D+ weights only, keep norms/biases in original dtype)
    fp8_dict = {}
    for key, tensor in state_dict.items():
        if tensor.is_floating_point() and tensor.ndim >= 2:
            fp8_dict[key] = tensor.to(torch.float8_e4m3fn)
        else:
            fp8_dict[key] = tensor

    # Save FP8 file
    save_file(fp8_dict, str(fp8_file))
    size_gb = fp8_file.stat().st_size / 1024**3
    print(f"[OK] {label} FP8 saved: {fp8_file.name} ({size_gb:.1f} GB)")
    del state_dict, fp8_dict

    # Delete BF16 shards and index files
    deleted = 0
    for f in bf16_files:
        if f.exists():
            f.unlink()
            deleted += 1
    print(f"[OK] Deleted {deleted} BF16 file(s) from {transformer_dir.name}/")


# ---------------------------------------------------------------------------
# Klein downloads (official FP8)
# ---------------------------------------------------------------------------
def download_klein_model(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    dest = model_dir / KLEIN_MODEL_FILE
    if dest.exists():
        print(f"[OK] Klein 4B FP8 already exists: {dest}")
        return
    print(f"[DL] Downloading Klein 4B FP8 -> {dest}")
    hf_hub_download(
        KLEIN_4B_FP8_REPO,
        filename=KLEIN_MODEL_FILE,
        local_dir=str(model_dir),
    )
    print(f"[OK] Klein 4B FP8 downloaded")


def download_klein_base_model(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    dest = model_dir / KLEIN_BASE_MODEL_FILE
    if dest.exists():
        print(f"[OK] Klein Base 4B FP8 already exists: {dest}")
        return
    print(f"[DL] Downloading Klein Base 4B FP8 -> {dest}")
    hf_hub_download(
        KLEIN_BASE_4B_FP8_REPO,
        filename=KLEIN_BASE_MODEL_FILE,
        local_dir=str(model_dir),
    )
    print(f"[OK] Klein Base 4B FP8 downloaded")


def download_klein_ae(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    dest = model_dir / KLEIN_AE_FILE
    if dest.exists():
        print(f"[OK] Klein AE already exists: {dest}")
        return
    print(f"[DL] Downloading Klein AE -> {dest}")
    hf_hub_download(
        KLEIN_AE_REPO,
        filename=KLEIN_AE_FILE,
        local_dir=str(model_dir),
    )
    print(f"[OK] Klein AE downloaded")


def download_klein_text_encoder():
    """Pre-download Klein text encoder (Qwen3-4B-FP8) to HF cache."""
    try:
        snapshot_download(
            KLEIN_TEXT_ENCODER_REPO,
            ignore_patterns=["*.md", ".gitattributes", "*.gguf"],
        )
        print(f"[OK] Klein Text Encoder (Qwen3-4B-FP8) downloaded")
    except Exception as e:
        print(f"[WARN] Klein Text Encoder download failed: {e}")
        print(f"  Will be auto-downloaded on first use.")


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------
def check_status(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    print(f"Model directory: {model_dir}")
    print()

    # Z-Image: check for FP8 transformer inside model folder
    for label, subdir in [("Z-Image-Turbo", ZIMAGE_TURBO_DIR), ("Z-Image-Base", ZIMAGE_BASE_DIR)]:
        model_path = model_dir / subdir
        fp8_file = model_path / "transformer" / FP8_TRANSFORMER_FILENAME
        if fp8_file.exists():
            size = fp8_file.stat().st_size / 1024**3
            print(f"  [OK] {label} (FP8, {size:.1f} GB)")
        elif model_path.exists():
            print(f"  [WARN] {label} (BF16 — needs FP8 conversion)")
        else:
            print(f"  [MISSING] {label}")

    # Klein
    for label, fname in [
        ("Klein 4B (FP8)", KLEIN_MODEL_FILE),
        ("Klein Base 4B (FP8)", KLEIN_BASE_MODEL_FILE),
        ("Klein AE", KLEIN_AE_FILE),
    ]:
        path = model_dir / fname
        if path.exists():
            size = path.stat().st_size / 1024**3
            print(f"  [OK] {label} ({size:.1f} GB)")
        else:
            print(f"  [MISSING] {label}")

    # Klein text encoder cache
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        qwen_cached = any("Qwen3-4B-FP8" in str(r.repo_id) for r in cache_info.repos)
        if qwen_cached:
            print(f"  [OK] Klein Text Encoder (Qwen3-4B-FP8) cached")
        else:
            print(f"  [MISSING] Klein Text Encoder (Qwen3-4B-FP8)")
    except Exception:
        print(f"  [?] Klein Text Encoder — cannot check cache")


# ---------------------------------------------------------------------------
# Download all + convert
# ---------------------------------------------------------------------------
def download_all(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / "loras" / "zimage").mkdir(parents=True, exist_ok=True)
    (model_dir / "loras" / "klein").mkdir(parents=True, exist_ok=True)

    download_zimage_turbo(model_dir)
    download_zimage_base(model_dir)
    download_klein_model(model_dir)
    download_klein_base_model(model_dir)
    download_klein_ae(model_dir)
    download_klein_text_encoder()

    print()
    check_status(model_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_status()
    else:
        download_all()
