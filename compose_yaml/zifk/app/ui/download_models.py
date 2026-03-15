"""Model download, FP8 conversion, and status checker for ZIFK.

Downloads Z-Image Turbo/Base (BF16 → FP8 convert) and FLUX.2 Klein 4B (official FP8).
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
    ZIMAGE_BASE_FP8_FILE,
    ZIMAGE_BASE_REPO,
    ZIMAGE_TURBO_DIR,
    ZIMAGE_TURBO_FP8_FILE,
    ZIMAGE_TURBO_REPO,
)


# ---------------------------------------------------------------------------
# Z-Image downloads (BF16 diffusers format — for config/VAE/text_encoder + FP8 conversion)
# ---------------------------------------------------------------------------
def download_zimage_turbo(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    dest = model_dir / ZIMAGE_TURBO_DIR
    if dest.exists() and any(dest.rglob("*.safetensors")):
        print(f"[OK] Z-Image-Turbo (BF16) already exists: {dest}")
        return
    print(f"[DL] Downloading Z-Image-Turbo (BF16) -> {dest}")
    snapshot_download(
        ZIMAGE_TURBO_REPO,
        local_dir=str(dest),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print(f"[OK] Z-Image-Turbo (BF16) downloaded")


def download_zimage_base(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    dest = model_dir / ZIMAGE_BASE_DIR
    if dest.exists() and any(dest.rglob("*.safetensors")):
        print(f"[OK] Z-Image-Base (BF16) already exists: {dest}")
        return
    print(f"[DL] Downloading Z-Image-Base (BF16) -> {dest}")
    snapshot_download(
        ZIMAGE_BASE_REPO,
        local_dir=str(dest),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print(f"[OK] Z-Image-Base (BF16) downloaded")


# ---------------------------------------------------------------------------
# Z-Image FP8 conversion (BF16 transformer → FP8 e4m3fn single file)
# ---------------------------------------------------------------------------
def convert_zimage_fp8(model_dir: Path | None = None, model_type: str = "turbo"):
    """Convert Z-Image transformer weights from BF16 to FP8 (e4m3fn).

    Reads all transformer safetensors from the BF16 folder,
    casts to float8_e4m3fn, saves as a single file.
    VAE/text_encoder/scheduler remain in original precision.
    """
    import torch
    from safetensors.torch import load_file, save_file

    model_dir = model_dir or MODEL_DIR

    if model_type == "turbo":
        src_dir = model_dir / ZIMAGE_TURBO_DIR / "transformer"
        dst_file = model_dir / ZIMAGE_TURBO_FP8_FILE
        label = "Z-Image-Turbo"
    else:
        src_dir = model_dir / ZIMAGE_BASE_DIR / "transformer"
        dst_file = model_dir / ZIMAGE_BASE_FP8_FILE
        label = "Z-Image-Base"

    if dst_file.exists():
        print(f"[OK] {label} FP8 already exists: {dst_file}")
        return

    if not src_dir.exists():
        print(f"[SKIP] {label} BF16 not found — download first")
        return

    print(f"[CVT] Converting {label} transformer to FP8...")

    # Load all transformer safetensors (may be sharded)
    state_dict = {}
    index_files = list(src_dir.glob("*.safetensors.index.json"))
    if index_files:
        import json
        with open(index_files[0]) as f:
            index = json.load(f)
        shard_files = set(index.get("weight_map", {}).values())
        for shard in shard_files:
            shard_dict = load_file(str(src_dir / shard), device="cpu")
            state_dict.update(shard_dict)
    else:
        for sf in src_dir.glob("*.safetensors"):
            shard_dict = load_file(str(sf), device="cpu")
            state_dict.update(shard_dict)

    if not state_dict:
        print(f"[ERR] No safetensors files found in {src_dir}")
        return

    # Cast to FP8 e4m3fn
    fp8_dict = {}
    for key, tensor in state_dict.items():
        if tensor.is_floating_point() and tensor.ndim >= 2:
            fp8_dict[key] = tensor.to(torch.float8_e4m3fn)
        else:
            # Keep non-floating (int) and 1D tensors (norms, biases) in original dtype
            fp8_dict[key] = tensor

    save_file(fp8_dict, str(dst_file))
    size_gb = dst_file.stat().st_size / 1024**3
    print(f"[OK] {label} FP8 saved: {dst_file} ({size_gb:.1f} GB)")
    del state_dict, fp8_dict


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

    checks = [
        ("Z-Image-Turbo (BF16)", model_dir / ZIMAGE_TURBO_DIR, True),
        ("Z-Image-Turbo (FP8)", model_dir / ZIMAGE_TURBO_FP8_FILE, False),
        ("Z-Image-Base (BF16)", model_dir / ZIMAGE_BASE_DIR, True),
        ("Z-Image-Base (FP8)", model_dir / ZIMAGE_BASE_FP8_FILE, False),
        ("Klein 4B (FP8)", model_dir / KLEIN_MODEL_FILE, False),
        ("Klein Base 4B (FP8)", model_dir / KLEIN_BASE_MODEL_FILE, False),
        ("Klein AE", model_dir / KLEIN_AE_FILE, False),
    ]

    for name, path, is_dir in checks:
        if is_dir:
            ok = path.exists() and any(path.rglob("*.safetensors"))
        else:
            ok = path.exists()
        status = "OK" if ok else "MISSING"
        size = ""
        if ok and not is_dir:
            size = f" ({path.stat().st_size / 1024**3:.1f} GB)"
        elif ok and is_dir:
            total = sum(f.stat().st_size for f in path.rglob("*.safetensors"))
            size = f" ({total / 1024**3:.1f} GB)"
        print(f"  [{status}] {name}{size}")

    # Klein text encoder cache check
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        qwen_cached = any("Qwen3-4B-FP8" in str(r.repo_id) for r in cache_info.repos)
        if qwen_cached:
            print(f"  [OK] Klein Text Encoder (Qwen3-4B-FP8) cached")
        else:
            print(f"  [MISSING] Klein Text Encoder (Qwen3-4B-FP8)")
    except Exception:
        print(f"  [?] Klein Text Encoder (Qwen3-4B-FP8) — cannot check cache")


# ---------------------------------------------------------------------------
# Download all + convert
# ---------------------------------------------------------------------------
def download_all(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # LoRA dirs
    (model_dir / "loras" / "zimage").mkdir(parents=True, exist_ok=True)
    (model_dir / "loras" / "klein").mkdir(parents=True, exist_ok=True)

    # Z-Image: download BF16 + convert to FP8
    download_zimage_turbo(model_dir)
    convert_zimage_fp8(model_dir, "turbo")
    download_zimage_base(model_dir)
    convert_zimage_fp8(model_dir, "base")

    # Klein: download official FP8
    download_klein_model(model_dir)
    download_klein_base_model(model_dir)
    download_klein_ae(model_dir)
    download_klein_text_encoder()

    print()
    check_status(model_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_status()
    elif len(sys.argv) > 1 and sys.argv[1] == "convert":
        # Convert only (BF16 already downloaded)
        convert_zimage_fp8(MODEL_DIR, "turbo")
        convert_zimage_fp8(MODEL_DIR, "base")
    else:
        download_all()
