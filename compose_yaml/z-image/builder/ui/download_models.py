"""Download Z-Image model files from HuggingFace.

Downloads Turbo and Base models in diffusers format.
Sequential downloads (ISP QoS limits parallel connections).

Usage:
    python download_models.py                # Download all required models
    python download_models.py --only turbo   # Download specific model
    python download_models.py --status       # Check download status
"""

import argparse
import sys
from pathlib import Path

from config import MODEL_DIR, TURBO_REPO, BASE_REPO

TURBO_DIR = "Z-Image-Turbo"
BASE_DIR = "Z-Image"
LORAS_DIR = "loras"


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f}GB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.0f}MB"
    return f"{size_bytes / 1024:.0f}KB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def check_status() -> None:
    print(f"Model directory: {MODEL_DIR}\n")

    models = [
        (TURBO_DIR, "Z-Image-Turbo (diffusers, ~33GB)"),
        (BASE_DIR, "Z-Image Base (diffusers, ~21GB)"),
    ]
    for dirname, desc in models:
        path = MODEL_DIR / dirname
        if path.exists() and any(path.rglob("*.safetensors")):
            size = dir_size(path)
            print(f"  [  OK  ] {dirname}/ ({format_size(size)}) — {desc}")
        else:
            print(f"  [MISSING] {dirname}/ — {desc}")

    loras = MODEL_DIR / LORAS_DIR
    if loras.exists():
        lora_files = list(loras.glob("*.safetensors"))
        if lora_files:
            print(f"\n  --- LoRAs ({len(lora_files)} files) ---")
            for f in sorted(lora_files):
                print(f"  [  OK  ] loras/{f.name} ({format_size(f.stat().st_size)})")
        else:
            print(f"\n  --- LoRAs (empty) ---")
    else:
        print(f"\n  --- LoRAs (not created) ---")


def download_model(repo_id: str, local_dir: str) -> bool:
    from huggingface_hub import snapshot_download

    dest = MODEL_DIR / local_dir
    if dest.exists() and any(dest.rglob("*.safetensors")):
        size = dir_size(dest)
        print(f"  Already exists: {local_dir}/ ({format_size(size)})")
        return True

    print(f"  Downloading: {repo_id} → {dest}...")
    try:
        snapshot_download(
            repo_id,
            local_dir=str(dest),
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        size = dir_size(dest)
        print(f"  Done: {local_dir}/ ({format_size(size)})")
        return True
    except Exception as e:
        print(f"  FAILED: {repo_id} — {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Z-Image models")
    parser.add_argument("--status", action="store_true", help="Check download status")
    parser.add_argument("--only", type=str, help="Download specific: turbo, base")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()

    global MODEL_DIR
    MODEL_DIR = Path(args.model_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / LORAS_DIR).mkdir(parents=True, exist_ok=True)

    if args.status:
        check_status()
        return

    if args.only:
        key = args.only.lower()
        if key == "turbo":
            sys.exit(0 if download_model(TURBO_REPO, TURBO_DIR) else 1)
        elif key == "base":
            sys.exit(0 if download_model(BASE_REPO, BASE_DIR) else 1)
        else:
            print(f"Unknown: {args.only}")
            print("Available: turbo, base")
            sys.exit(1)

    print(f"Downloading models to: {MODEL_DIR}\n")
    results = {}

    # 1. Turbo
    print("--- Z-Image-Turbo ---")
    results["turbo"] = download_model(TURBO_REPO, TURBO_DIR)

    # 2. Base
    print("\n--- Z-Image Base ---")
    results["base"] = download_model(BASE_REPO, BASE_DIR)

    print("\n" + "=" * 50)
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
