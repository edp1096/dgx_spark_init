"""Model download and status checker for ZIT.

Downloads Z-Image Turbo (BF16), ControlNet Union, and preprocessor weights.
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
    HED_FILE,
    HED_URL,
    LORAS_DIR,
    TRAINING_ADAPTER_DIR,
    TRAINING_ADAPTER_FILENAME,
    TRAINING_ADAPTER_URL,
    TRANSLATOR_DIR,
    TRANSLATOR_REPO,
    MODEL_DIR,
    PREPROCESSORS_DIR,
    ZOEDEPTH_FILE,
    ZOEDEPTH_URL,
    ZIMAGE_TURBO_DIR,
    ZIMAGE_TURBO_REPO,
)

def download_zimage_turbo(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    dest = model_dir / ZIMAGE_TURBO_DIR
    if dest.exists() and any(dest.rglob("*.safetensors")):
        print(f"[OK] Z-Image-Turbo (BF16) already exists")
        return
    print(f"[DL] Downloading Z-Image-Turbo -> {dest}")
    snapshot_download(
        ZIMAGE_TURBO_REPO,
        local_dir=str(dest),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print(f"[OK] Z-Image-Turbo downloaded")


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
# Training adapter download (de-distillation)
# ---------------------------------------------------------------------------
def download_training_adapter(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    adapter_dir = model_dir / TRAINING_ADAPTER_DIR
    adapter_dir.mkdir(parents=True, exist_ok=True)
    dest = adapter_dir / TRAINING_ADAPTER_FILENAME
    if dest.exists():
        print(f"[OK] Training adapter already exists")
        return
    _download_url(TRAINING_ADAPTER_URL, dest)


# ---------------------------------------------------------------------------
# Translator model download
# ---------------------------------------------------------------------------
def download_translator(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    dest = model_dir / TRANSLATOR_DIR
    if dest.exists() and (any(dest.rglob("*.safetensors")) or any(dest.rglob("*.bin"))):
        print(f"[OK] {TRANSLATOR_DIR} already exists")
        return
    print(f"[DL] Downloading {TRANSLATOR_DIR} -> {dest}")
    snapshot_download(
        TRANSLATOR_REPO,
        local_dir=str(dest),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print(f"[OK] {TRANSLATOR_DIR} downloaded")


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------
def check_status(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    print(f"Model directory: {model_dir}")
    print()

    # Z-Image Turbo
    turbo_path = model_dir / ZIMAGE_TURBO_DIR
    if turbo_path.exists() and any(turbo_path.rglob("*.safetensors")):
        size = sum(f.stat().st_size for f in turbo_path.rglob("*.safetensors")) / 1024**3
        print(f"  [OK] Z-Image-Turbo (BF16, {size:.1f} GB)")
    else:
        print(f"  [MISSING] Z-Image-Turbo")

    # ControlNet Union
    cn_path = model_dir / CONTROLNET_DIR / CONTROLNET_FILENAME
    if cn_path.exists():
        size = cn_path.stat().st_size / 1024**3
        print(f"  [OK] ControlNet Union ({size:.1f} GB)")
    else:
        print(f"  [MISSING] ControlNet Union")

    # Translator
    translator_path = model_dir / TRANSLATOR_DIR
    if translator_path.exists() and (any(translator_path.rglob("*.safetensors")) or any(translator_path.rglob("*.bin"))):
        size = sum(f.stat().st_size for f in translator_path.rglob("*.safetensors")) + sum(f.stat().st_size for f in translator_path.rglob("*.bin"))
        print(f"  [OK] {TRANSLATOR_DIR} ({size / 1024**3:.1f} GB)")
    else:
        print(f"  [MISSING] {TRANSLATOR_DIR}")

    # Training adapter
    adapter_path = model_dir / TRAINING_ADAPTER_DIR / TRAINING_ADAPTER_FILENAME
    if adapter_path.exists():
        size = adapter_path.stat().st_size / 1024**2
        print(f"  [OK] Training adapter ({size:.0f} MB)")
    else:
        print(f"  [MISSING] Training adapter")

    # Preprocessors
    prep_dir = model_dir / PREPROCESSORS_DIR
    for fname in [DWPOSE_DET_FILE, DWPOSE_POSE_FILE, ZOEDEPTH_FILE, HED_FILE]:
        path = prep_dir / fname
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
    download_training_adapter(model_dir)
    download_preprocessors(model_dir)
    download_translator(model_dir)

    print()
    check_status(model_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_status()
    else:
        download_all()
