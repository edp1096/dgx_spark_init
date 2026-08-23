"""Klein (FLUX) Distilled FP8 — end-to-end image generation test."""
import sys
sys.path.insert(0, "flux2/src")
sys.path.insert(0, "app/ui")

import os
import torch
import numpy as np
import logging
from einops import rearrange
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = "/root/.cache/huggingface/hub/zifk"
DEVICE = "cuda"


def main():
    from zifk_config import KLEIN_MODEL_FILE, KLEIN_AE_FILE, KLEIN_DISTILLED
    from pipeline_manager import _Q8_AVAILABLE, _patch_transformer_q8, _load_fp8_weight_scales

    os.environ["KLEIN_4B_MODEL_PATH"] = f"{MODEL_DIR}/{KLEIN_MODEL_FILE}"
    os.environ["AE_MODEL_PATH"] = f"{MODEL_DIR}/{KLEIN_AE_FILE}"

    from flux2.util import load_ae, load_flow_model, load_text_encoder
    from flux2.sampling import batched_prc_img, batched_prc_txt, denoise, get_schedule, scatter_ids

    log.info("Loading Klein Distilled flow model...")
    model = load_flow_model(KLEIN_DISTILLED, device=DEVICE)
    model.eval()

    if any(p.dtype == torch.float8_e4m3fn for p in model.parameters()) and _Q8_AVAILABLE:
        weight_scales = _load_fp8_weight_scales(f"{MODEL_DIR}/{KLEIN_MODEL_FILE}")
        _patch_transformer_q8(model, weight_scales=weight_scales)
        model.dtype = torch.bfloat16
        log.info(f"FP8 patch applied ({len(weight_scales)} weight scales)")

    log.info("Loading text encoder + autoencoder...")
    text_encoder = load_text_encoder(KLEIN_DISTILLED, device=DEVICE)
    text_encoder.eval()
    ae = load_ae(KLEIN_DISTILLED)
    ae.eval()

    log.info("Generating (4 steps, guidance 1.0)...")
    with torch.no_grad():
        ctx = text_encoder(["a beautiful sunset over the ocean, vibrant colors, photorealistic"]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        x = torch.randn(1, 128, 64, 64, generator=torch.Generator(DEVICE).manual_seed(42),
                         dtype=torch.bfloat16, device=DEVICE)
        x, x_ids = batched_prc_img(x)
        timesteps = get_schedule(4, x.shape[1])

        x = denoise(model, x, x_ids, ctx, ctx_ids, timesteps=timesteps, guidance=1.0)
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = ae.decode(x).float()

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    arr = ((x + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()

    log.info(f"Image: {arr.shape} min={arr.min()} max={arr.max()} mean={arr.mean():.2f}")
    assert arr.max() > 10, "Image is too dark or black"
    Image.fromarray(arr).save("tests/output_klein.png")
    log.info("PASS — saved to tests/output_klein.png")

    del model, text_encoder, ae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
