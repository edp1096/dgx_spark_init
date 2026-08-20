"""Convert LTX-2.3 BF16 checkpoint to FP8 (float8_e4m3fn) on CPU.

Converts transformer block weights to FP8 and saves as a new safetensors file.
This avoids the peak memory spike of BF16→FP8 conversion during model loading.

Usage:
    python convert_fp8.py                          # Convert both dev and distilled
    python convert_fp8.py --only distilled         # Convert specific checkpoint
"""

import argparse
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from config import MODEL_DIR

# Same prefixes/suffixes as TRANSFORMER_LINEAR_DOWNCAST_MAP in fp8_cast.py
SUFFIXES = [
    ".to_q.weight", ".to_q.bias",
    ".to_k.weight", ".to_k.bias",
    ".to_v.weight", ".to_v.bias",
    ".to_out.0.weight", ".to_out.0.bias",
    "ff.net.0.proj.weight", "ff.net.0.proj.bias",
    "ff.net.2.weight", "ff.net.2.bias",
]


def should_quantize(key: str) -> bool:
    if "transformer_blocks." not in key:
        return False
    return any(key.endswith(s) for s in SUFFIXES)


def convert_checkpoint(src_path: Path, dst_path: Path) -> None:
    print(f"Loading: {src_path.name} ({src_path.stat().st_size / 1024**3:.1f}GB)")
    t0 = time.time()
    sd = load_file(str(src_path), device="cpu")
    print(f"  Loaded in {time.time() - t0:.1f}s ({len(sd)} keys)")

    # Preserve original metadata (contains model config)
    with safe_open(str(src_path), framework="pt") as f:
        metadata = f.metadata() or {}

    converted = 0
    for key in sd:
        if should_quantize(key):
            sd[key] = sd[key].to(dtype=torch.float8_e4m3fn)
            converted += 1

    print(f"  Converted {converted} tensors to FP8")
    print(f"  Saving: {dst_path.name}...")
    t0 = time.time()
    save_file(sd, str(dst_path), metadata=metadata)
    print(f"  Saved in {time.time() - t0:.1f}s ({dst_path.stat().st_size / 1024**3:.1f}GB)")


def main():
    global MODEL_DIR
    parser = argparse.ArgumentParser(description="Convert LTX checkpoints to FP8")
    parser.add_argument("--only", choices=["dev", "distilled"], help="Convert specific checkpoint")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    MODEL_DIR = Path(args.model_dir)

    checkpoints = {
        "dev": "ltx-2.3-22b-dev.safetensors",
        "distilled": "ltx-2.3-22b-distilled.safetensors",
    }

    targets = [args.only] if args.only else list(checkpoints.keys())

    for name in targets:
        src = MODEL_DIR / checkpoints[name]
        dst = MODEL_DIR / checkpoints[name].replace(".safetensors", "-fp8.safetensors")

        if dst.exists():
            print(f"Already exists: {dst.name} ({dst.stat().st_size / 1024**3:.1f}GB)")
            continue

        if not src.exists():
            print(f"Source not found: {src.name}")
            continue

        convert_checkpoint(src, dst)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
