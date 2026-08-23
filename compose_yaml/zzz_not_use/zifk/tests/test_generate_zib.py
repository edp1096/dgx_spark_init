"""Z-Image Base BF16 — end-to-end image generation test.

Z-Image Base uses BF16 (not FP8) because FP8 quantization error accumulates
over 28 denoising steps, producing noisy output. ComfyUI/Forge also use
full_precision_matrix_mult for Z-Image FP8 (BF16 matmul fallback).
"""
import sys
sys.path.insert(0, "Z-Image/src")
sys.path.insert(0, "app/ui")

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = "/root/.cache/huggingface/hub/zifk"
DEVICE = "cuda"


def main():
    from utils import load_from_local_dir

    log.info("Loading Z-Image Base (BF16)...")
    components = load_from_local_dir(
        f"{MODEL_DIR}/Z-Image", device=DEVICE, dtype=torch.bfloat16,
        verbose=True, compile=False,
    )

    log.info("Generating (28 steps, CFG 3.5)...")
    from zimage import generate
    images = generate(
        **components,
        prompt="a beautiful sunset over the ocean, vibrant colors, photorealistic",
        height=1024, width=1024,
        num_inference_steps=28, guidance_scale=3.5,
        generator=torch.Generator(DEVICE).manual_seed(42),
    )

    arr = np.array(images[0])
    log.info(f"Image: {arr.shape} min={arr.min()} max={arr.max()} mean={arr.mean():.2f}")
    assert arr.max() > 10, "Image is too dark or black"
    images[0].save("tests/output_zib.png")
    log.info("PASS — saved to tests/output_zib.png")

    del components
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
