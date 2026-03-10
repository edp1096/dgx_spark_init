"""Download LTX-2.3 model files from HuggingFace (FP8 mode).

Downloads dev-fp8 from official repo, distilled BF16 + auto-converts to FP8.
Downloads sequentially (ISP QoS limits parallel connections).

Usage:
    python download_models.py                    # Download all required models
    python download_models.py --only dev-fp8     # Download specific file
    python download_models.py --status           # Check download status
"""

import argparse
import sys
from pathlib import Path

from config import MODEL_DIR

# (filename, repo_id, description)
DOWNLOADS = {
    "dev-fp8": ("ltx-2.3-22b-dev-fp8.safetensors", "Lightricks/LTX-2.3-fp8", "Official dev FP8 (~29GB)"),
    "distilled": ("ltx-2.3-22b-distilled.safetensors", "Lightricks/LTX-2.3", "Distilled BF16 (~44GB, for FP8 conversion)"),
    "upscaler": ("ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "Lightricks/LTX-2.3", "2x Spatial upscaler (~950MB)"),
    "lora": ("ltx-2.3-22b-distilled-lora-384.safetensors", "Lightricks/LTX-2.3", "Distilled LoRA (~7.1GB)"),
}

GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f}GB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.0f}MB"
    return f"{size_bytes / 1024:.0f}KB"


def check_status() -> None:
    print(f"Model directory: {MODEL_DIR}\n")

    # Required FP8 files
    required = [
        "ltx-2.3-22b-dev-fp8.safetensors",
        "ltx-2.3-22b-distilled-fp8.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        "ltx-2.3-22b-distilled-lora-384.safetensors",
        "gemma-3-12b-it-qat-q4_0-unquantized",
    ]
    for fname in required:
        path = MODEL_DIR / fname
        if path.exists():
            if path.is_dir():
                print(f"  [  OK  ] {fname}/ (dir)")
            else:
                print(f"  [  OK  ] {fname} ({format_size(path.stat().st_size)})")
        else:
            print(f"  [MISSING] {fname}")


def download_file(filename: str, repo_id: str) -> bool:
    from huggingface_hub import hf_hub_download

    path = MODEL_DIR / filename
    if path.exists():
        print(f"  Already exists: {filename} ({format_size(path.stat().st_size)})")
        return True

    print(f"  Downloading: {filename} from {repo_id}...")
    try:
        hf_hub_download(repo_id, filename, local_dir=str(MODEL_DIR))
        print(f"  Done: {filename}")
        return True
    except Exception as e:
        print(f"  FAILED: {filename} — {e}")
        return False


def download_gemma() -> bool:
    from huggingface_hub import snapshot_download

    gemma_path = MODEL_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"
    if gemma_path.exists():
        print(f"  Already exists: gemma-3-12b-it-qat-q4_0-unquantized/")
        return True

    print(f"  Downloading: {GEMMA_REPO} (gated — HF login required)...")
    try:
        snapshot_download(GEMMA_REPO, local_dir=str(gemma_path))
        print(f"  Done: gemma-3-12b-it-qat-q4_0-unquantized/")
        return True
    except Exception as e:
        print(f"  FAILED: gemma — {e}")
        return False


def convert_distilled_fp8() -> bool:
    """Download distilled BF16, convert to FP8, delete BF16."""
    dst = MODEL_DIR / "ltx-2.3-22b-distilled-fp8.safetensors"
    if dst.exists():
        print(f"  Already exists: {dst.name} ({format_size(dst.stat().st_size)})")
        return True

    src = MODEL_DIR / "ltx-2.3-22b-distilled.safetensors"

    # Download BF16 if not present
    if not src.exists():
        print("  Downloading distilled BF16 for FP8 conversion...")
        ok = download_file(src.name, "Lightricks/LTX-2.3")
        if not ok:
            return False

    # Convert
    print("  Converting distilled BF16 → FP8...")
    try:
        import time

        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        t0 = time.time()
        sd = load_file(str(src), device="cpu")

        with safe_open(str(src), framework="pt") as f:
            metadata = f.metadata() or {}

        SUFFIXES = [
            ".to_q.weight", ".to_q.bias", ".to_k.weight", ".to_k.bias",
            ".to_v.weight", ".to_v.bias", ".to_out.0.weight", ".to_out.0.bias",
            "ff.net.0.proj.weight", "ff.net.0.proj.bias",
            "ff.net.2.weight", "ff.net.2.bias",
        ]
        converted = 0
        for key in sd:
            if "transformer_blocks." in key and any(key.endswith(s) for s in SUFFIXES):
                sd[key] = sd[key].to(dtype=torch.float8_e4m3fn)
                converted += 1

        save_file(sd, str(dst), metadata=metadata)
        print(f"  Converted {converted} tensors in {time.time() - t0:.0f}s → {dst.name} ({format_size(dst.stat().st_size)})")

        # Delete BF16 source to save space
        src.unlink()
        print(f"  Deleted BF16 source: {src.name}")
        return True
    except Exception as e:
        print(f"  FAILED: conversion — {e}")
        return False


def main() -> None:
    global MODEL_DIR

    parser = argparse.ArgumentParser(description="Download LTX-2.3 models (FP8)")
    parser.add_argument("--status", action="store_true", help="Check download status")
    parser.add_argument("--only", type=str, help="Download specific: dev-fp8, distilled-fp8, upscaler, lora, gemma")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()

    MODEL_DIR = Path(args.model_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        check_status()
        return

    if args.only:
        key = args.only.lower().replace("-", "_")
        if key == "gemma":
            sys.exit(0 if download_gemma() else 1)
        elif key == "distilled_fp8":
            sys.exit(0 if convert_distilled_fp8() else 1)
        elif key in {"dev_fp8", "upscaler", "lora"}:
            fname, repo, _ = DOWNLOADS[key.replace("_", "-")]
            sys.exit(0 if download_file(fname, repo) else 1)
        else:
            print(f"Unknown: {args.only}")
            print("Available: dev-fp8, distilled-fp8, upscaler, lora, gemma")
            sys.exit(1)

    # Download all sequentially
    print(f"Downloading all models to: {MODEL_DIR}\n")
    results = {}

    # 1. Dev FP8 (official)
    fname, repo, _ = DOWNLOADS["dev-fp8"]
    results["dev-fp8"] = download_file(fname, repo)

    # 2. Distilled FP8 (download BF16 + convert)
    results["distilled-fp8"] = convert_distilled_fp8()

    # 3. Upscaler
    fname, repo, _ = DOWNLOADS["upscaler"]
    results["upscaler"] = download_file(fname, repo)

    # 4. LoRA
    fname, repo, _ = DOWNLOADS["lora"]
    results["lora"] = download_file(fname, repo)

    # 5. Gemma
    results["gemma"] = download_gemma()

    print("\n" + "=" * 50)
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
