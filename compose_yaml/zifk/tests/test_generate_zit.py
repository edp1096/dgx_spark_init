"""Z-Image Turbo FP8 — end-to-end image generation test."""
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
    from safetensors.torch import load_file
    from pipeline_manager import _patch_transformer_q8, _load_fp8_weight_scales

    log.info("Loading Z-Image Turbo...")
    components = load_from_local_dir(
        f"{MODEL_DIR}/Z-Image-Turbo", device=DEVICE, dtype=torch.bfloat16,
        verbose=True, compile=False,
    )

    fp8_file = f"{MODEL_DIR}/Z-Image-Turbo/transformer/model_fp8.safetensors"
    fp8_state = load_file(fp8_file, device=DEVICE)
    components["transformer"].load_state_dict(fp8_state, strict=False, assign=True)
    del fp8_state
    torch.cuda.empty_cache()

    weight_scales = _load_fp8_weight_scales(fp8_file)
    _patch_transformer_q8(components["transformer"], weight_scales=weight_scales)
    components["transformer"].dtype = torch.bfloat16

    log.info("Generating (4 steps, no CFG)...")
    from zimage import generate
    images = generate(
        **components,
        prompt="a beautiful sunset over the ocean, vibrant colors, photorealistic",
        height=1024, width=1024,
        num_inference_steps=4, guidance_scale=0.0,
        generator=torch.Generator(DEVICE).manual_seed(42),
    )

    arr = np.array(images[0])
    log.info(f"Image: {arr.shape} min={arr.min()} max={arr.max()} mean={arr.mean():.2f}")
    assert arr.max() > 10, "Image is too dark or black"
    images[0].save("tests/output_zit.png")
    log.info("PASS — saved to tests/output_zit.png")

    del components
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
