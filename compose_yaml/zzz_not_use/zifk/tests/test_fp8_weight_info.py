"""FP8 weight analysis — check dtype distribution, scales, and ranges.

Usage:
    python tests/test_fp8_weight_info.py
"""
import torch
from safetensors.torch import load_file
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = "/root/.cache/huggingface/hub/zifk"

CHECKPOINTS = [
    ("Z-Image Turbo",    f"{MODEL_DIR}/Z-Image-Turbo/transformer/model_fp8.safetensors"),
    ("Klein Distilled",  f"{MODEL_DIR}/flux-2-klein-4b-fp8.safetensors"),
    ("Klein Base",       f"{MODEL_DIR}/flux-2-klein-base-4b-fp8.safetensors"),
]


def analyze(label, path):
    log.info(f"\n{'=' * 60}")
    log.info(f"  {label}")
    log.info(f"  {path}")
    log.info(f"{'=' * 60}")

    try:
        sd = load_file(path, device="cpu")
    except FileNotFoundError:
        log.warning(f"  File not found — skipping")
        return

    dtypes = {}
    for k, t in sd.items():
        d = str(t.dtype)
        dtypes[d] = dtypes.get(d, 0) + 1

    log.info(f"  Keys: {len(sd)}")
    for d, c in sorted(dtypes.items()):
        log.info(f"    {d}: {c}")

    scale_keys = [k for k in sd if "scale" in k.lower()]
    fp8_keys = [k for k in sd if sd[k].dtype == torch.float8_e4m3fn]

    log.info(f"  FP8 weights: {len(fp8_keys)}")
    log.info(f"  Scale keys: {len(scale_keys)}")

    if scale_keys:
        for k in scale_keys[:5]:
            t = sd[k]
            val = t.float().item() if t.numel() == 1 else f"shape={tuple(t.shape)}"
            log.info(f"    {k}: {val}")

    if fp8_keys:
        absmax_vals = []
        for k in fp8_keys[:5]:
            am = sd[k].float().abs().max().item()
            absmax_vals.append(am)
            log.info(f"    {k}: absmax={am:.1f}")
        if len(fp8_keys) > 5:
            all_absmax = [sd[k].float().abs().max().item() for k in fp8_keys]
            log.info(f"    ... overall FP8 absmax range: [{min(all_absmax):.1f}, {max(all_absmax):.1f}]")

    del sd


def main():
    for label, path in CHECKPOINTS:
        analyze(label, path)


if __name__ == "__main__":
    main()
