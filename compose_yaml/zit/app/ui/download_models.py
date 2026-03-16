"""Model download, FP8 conversion, and status checker for ZIT.

Downloads Z-Image Turbo (BF16 → in-place FP8 convert, delete BF16 shards),
ControlNet Union, preprocessor weights, and FaceSwap models.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from zit_config import (
    CONTROLNET_DIR,
    CONTROLNET_FILENAME,
    CONTROLNET_REPO,
    DWPOSE_DET_FILE,
    DWPOSE_DET_URL,
    DWPOSE_POSE_FILE,
    DWPOSE_POSE_URL,
    FACESWAP_DIR,
    HED_FILE,
    HED_URL,
    INSWAPPER_FILE,
    LORAS_DIR,
    MODEL_DIR,
    PREPROCESSORS_DIR,
    ZOEDEPTH_FILE,
    ZOEDEPTH_URL,
    ZIMAGE_TURBO_DIR,
    ZIMAGE_TURBO_REPO,
)

FP8_TRANSFORMER_FILENAME = "model_fp8.safetensors"


# ---------------------------------------------------------------------------
# Z-Image Turbo download (BF16 → in-place FP8 conversion)
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

    LINEAR_WEIGHT_SUFFIXES = (
        ".to_q.weight", ".to_k.weight", ".to_v.weight",
        ".to_out.0.weight",
        ".w1.weight", ".w2.weight", ".w3.weight",
        ".linear.weight",
        "adaLN_modulation.0.weight",
        "cap_embedder.0.weight",
        "x_embedder.weight",
        "mlp.0.weight", "mlp.2.weight",
    )

    fp8_dict = {}
    converted = 0
    for key, tensor in state_dict.items():
        is_linear_weight = (
            tensor.is_floating_point()
            and tensor.ndim == 2
            and any(key.endswith(s) for s in LINEAR_WEIGHT_SUFFIXES)
        )
        if is_linear_weight:
            w_float = tensor.float()
            w_absmax = w_float.abs().amax().clamp(min=1e-12)
            weight_scale = w_absmax / 448.0
            fp8_dict[key] = (w_float / weight_scale).to(torch.float8_e4m3fn)
            fp8_dict[key.replace(".weight", ".weight_scale")] = weight_scale
            converted += 1
        else:
            fp8_dict[key] = tensor

    print(f"  Converted {converted} Linear weights to FP8 (normalized)")

    save_file(fp8_dict, str(fp8_file))
    size_gb = fp8_file.stat().st_size / 1024**3
    print(f"[OK] {label} FP8 saved: {fp8_file.name} ({size_gb:.1f} GB)")
    del state_dict, fp8_dict

    deleted = 0
    for f in bf16_files:
        if f.exists():
            f.unlink()
            deleted += 1
    print(f"[OK] Deleted {deleted} BF16 file(s) from {transformer_dir.name}/")


# ---------------------------------------------------------------------------
# ControlNet Union download
# ---------------------------------------------------------------------------
def download_controlnet(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    cn_dir = model_dir / CONTROLNET_DIR
    cn_dir.mkdir(parents=True, exist_ok=True)
    dest = cn_dir / CONTROLNET_FILENAME
    if dest.exists():
        print(f"[OK] ControlNet Union already exists")
        return
    print(f"[DL] Downloading ControlNet Union -> {dest}")
    hf_hub_download(
        CONTROLNET_REPO,
        filename=CONTROLNET_FILENAME,
        local_dir=str(cn_dir),
    )
    print(f"[OK] ControlNet Union downloaded")


# ---------------------------------------------------------------------------
# Preprocessor weight downloads
# ---------------------------------------------------------------------------
def _download_url(url: str, dest: Path):
    """Download a file from URL."""
    import urllib.request
    print(f"[DL] Downloading {dest.name}...")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / 1024**2
    print(f"[OK] {dest.name} ({size_mb:.0f} MB)")


def download_preprocessors(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    prep_dir = model_dir / PREPROCESSORS_DIR
    prep_dir.mkdir(parents=True, exist_ok=True)

    downloads = [
        (DWPOSE_DET_URL, prep_dir / DWPOSE_DET_FILE),
        (DWPOSE_POSE_URL, prep_dir / DWPOSE_POSE_FILE),
        (ZOEDEPTH_URL, prep_dir / ZOEDEPTH_FILE),
        (HED_URL, prep_dir / HED_FILE),
    ]

    for url, dest in downloads:
        if dest.exists():
            print(f"[OK] {dest.name} already exists")
        else:
            _download_url(url, dest)


# ---------------------------------------------------------------------------
# FaceSwap model downloads (placeholder — URLs TBD)
# ---------------------------------------------------------------------------
def download_faceswap(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    fs_dir = model_dir / FACESWAP_DIR
    fs_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Phase 5 — add actual download URLs for SCRFD, ArcFace, inswapper
    for fname in [INSWAPPER_FILE]:
        dest = fs_dir / fname
        if dest.exists():
            print(f"[OK] {fname} already exists")
        else:
            print(f"[MISSING] {fname} — manual download required")


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------
def check_status(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    print(f"Model directory: {model_dir}")
    print()

    # Z-Image Turbo
    turbo_path = model_dir / ZIMAGE_TURBO_DIR
    turbo_fp8 = turbo_path / "transformer" / FP8_TRANSFORMER_FILENAME
    if turbo_fp8.exists():
        size = turbo_fp8.stat().st_size / 1024**3
        print(f"  [OK] Z-Image-Turbo (FP8, {size:.1f} GB)")
    elif turbo_path.exists():
        print(f"  [WARN] Z-Image-Turbo (BF16 — needs FP8 conversion)")
    else:
        print(f"  [MISSING] Z-Image-Turbo")

    # ControlNet Union
    cn_path = model_dir / CONTROLNET_DIR / CONTROLNET_FILENAME
    if cn_path.exists():
        size = cn_path.stat().st_size / 1024**3
        print(f"  [OK] ControlNet Union ({size:.1f} GB)")
    else:
        print(f"  [MISSING] ControlNet Union")

    # Preprocessors
    prep_dir = model_dir / PREPROCESSORS_DIR
    for fname in [DWPOSE_DET_FILE, DWPOSE_POSE_FILE, ZOEDEPTH_FILE, HED_FILE]:
        path = prep_dir / fname
        if path.exists():
            size = path.stat().st_size / 1024**2
            print(f"  [OK] {fname} ({size:.0f} MB)")
        else:
            print(f"  [MISSING] {fname}")

    # FaceSwap
    fs_dir = model_dir / FACESWAP_DIR
    for fname in [INSWAPPER_FILE]:
        path = fs_dir / fname
        if path.exists():
            size = path.stat().st_size / 1024**2
            print(f"  [OK] {fname} ({size:.0f} MB)")
        else:
            print(f"  [MISSING] {fname}")


# ---------------------------------------------------------------------------
# Download all
# ---------------------------------------------------------------------------
def download_all(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / LORAS_DIR).mkdir(parents=True, exist_ok=True)

    download_zimage_turbo(model_dir)
    download_controlnet(model_dir)
    download_preprocessors(model_dir)
    download_faceswap(model_dir)

    print()
    check_status(model_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_status()
    else:
        download_all()
