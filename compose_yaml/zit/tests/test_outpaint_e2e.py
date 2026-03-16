"""Outpaint end-to-end GPU test — loads pipeline and runs actual generation.

Simulates the full outpaint flow:
1. Load ZIT + ControlNet pipeline (same as worker.py)
2. Create a test image (solid color)
3. Expand canvas right by 128px
4. Create mask (expanded area = white)
5. Run inpaint pipeline on expanded canvas

Usage:
    cd /root/zit-ui && python tests/test_outpaint_e2e.py
"""
import sys
import os
import time
import logging
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test-outpaint")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "ui"))

import numpy as np
import torch
from PIL import Image


def step(msg):
    log.info("=" * 60)
    log.info("  %s", msg)
    log.info("=" * 60)


def main():
    DEVICE = "cuda"
    MODEL_DIR = "/root/.cache/huggingface/hub/zit"
    OUTPUT_DIR = Path("/tmp/zit-outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: Load pipeline (replicates pipeline_manager.load_zit)
    # ---------------------------------------------------------------
    step("Step 1: Loading ZIT pipeline")
    t0 = time.time()

    from zit_config import CONTROLNET_CONFIG, CONTROLNET_DIR, CONTROLNET_FILENAME, ZIMAGE_TURBO_DIR
    from pipeline_manager import _Q8_AVAILABLE, _patch_transformer_q8, _load_fp8_weight_scales

    model_path = Path(MODEL_DIR) / ZIMAGE_TURBO_DIR
    transformer_dir = model_path / "transformer"

    log.info("Loading transformer from %s ...", transformer_dir)
    from videox_models.z_image_transformer2d_control import ZImageControlTransformer2DModel
    transformer = ZImageControlTransformer2DModel.from_pretrained(
        str(transformer_dir),
        torch_dtype=torch.bfloat16,
        transformer_additional_kwargs=CONTROLNET_CONFIG,
    )
    transformer = transformer.to(DEVICE)
    transformer.eval()
    log.info("Transformer loaded (%.1fs)", time.time() - t0)

    # Load ControlNet adapter
    cn_path = Path(MODEL_DIR) / CONTROLNET_DIR / CONTROLNET_FILENAME
    if cn_path.exists():
        log.info("Loading ControlNet adapter from %s ...", cn_path)
        from safetensors.torch import load_file
        cn_state = load_file(str(cn_path), device=DEVICE)
        missing, unexpected = transformer.load_state_dict(cn_state, strict=False)
        del cn_state
        torch.cuda.empty_cache()
        log.info("ControlNet loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))
    else:
        log.error("ControlNet NOT FOUND at %s — inpaint requires it!", cn_path)
        return 1

    # FP8 patch
    fp8_file = transformer_dir / "model_fp8.safetensors"
    if fp8_file.exists() and _Q8_AVAILABLE:
        log.info("Applying FP8 q8_kernels patch...")
        from safetensors.torch import load_file
        fp8_state = load_file(str(fp8_file), device=DEVICE)
        transformer.load_state_dict(fp8_state, strict=False, assign=True)
        del fp8_state
        torch.cuda.empty_cache()
        weight_scales = _load_fp8_weight_scales(str(fp8_file))
        _patch_transformer_q8(transformer, weight_scales=weight_scales)
        log.info("FP8 patch applied")
    else:
        log.info("Using BF16 (FP8 not available: fp8_exists=%s, q8=%s)", fp8_file.exists(), _Q8_AVAILABLE)

    # VAE
    log.info("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        str(model_path / "vae"), torch_dtype=torch.float32,
    ).to(DEVICE)
    vae.eval()

    # Text encoder
    log.info("Loading text encoder...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    text_encoder = AutoModelForCausalLM.from_pretrained(
        str(model_path / "text_encoder"), torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    text_encoder.eval()

    # Tokenizer + scheduler
    log.info("Loading tokenizer + scheduler...")
    tokenizer_dir = model_path / "tokenizer"
    if not tokenizer_dir.exists():
        tokenizer_dir = model_path / "text_encoder"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler_dir = model_path / "scheduler"
    if scheduler_dir.exists():
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_dir))
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)

    # Build pipeline
    from videox_models.pipeline_z_image_control import ZImageControlPipeline
    pipeline = ZImageControlPipeline(
        transformer=transformer, vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, scheduler=scheduler,
    )

    log.info("Pipeline ready (total %.1fs)", time.time() - t0)
    log.info("  transformer.in_channels = %s", transformer.in_channels)
    log.info("  transformer.control_in_dim = %s", transformer.control_in_dim)

    # ---------------------------------------------------------------
    # Step 2: Create test image + outpaint canvas
    # ---------------------------------------------------------------
    step("Step 2: Preparing outpaint canvas")

    # Create a simple test image (gradient)
    orig_w, orig_h = 256, 256
    orig_arr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for y in range(orig_h):
        for x in range(orig_w):
            orig_arr[y, x] = [int(x / orig_w * 255), int(y / orig_h * 255), 128]
    orig_img = Image.fromarray(orig_arr)

    # Expand right by 128px (simulates _run_outpaint)
    expand_px = 128
    direction = ["Right"]
    pad = {"Left": 0, "Right": 0, "Up": 0, "Down": 0}
    for d in direction:
        pad[d] = expand_px

    new_w = orig_w + pad["Left"] + pad["Right"]
    new_h = orig_h + pad["Up"] + pad["Down"]

    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    canvas.paste(orig_img, (pad["Left"], pad["Up"]))

    mask = Image.new("L", (new_w, new_h), 255)
    mask_arr = np.array(mask)
    mask_arr[pad["Up"]:pad["Up"]+orig_h, pad["Left"]:pad["Left"]+orig_w] = 0
    mask = Image.fromarray(mask_arr)

    # Align to 16px
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16

    log.info("Original: %dx%d → Expanded: %dx%d (aligned)", orig_w, orig_h, new_w, new_h)
    log.info("Canvas size: %s, Mask size: %s", canvas.size, mask.size)

    # Save for inspection
    canvas.save(str(OUTPUT_DIR / "test_outpaint_canvas.png"))
    mask.save(str(OUTPUT_DIR / "test_outpaint_mask.png"))
    log.info("Saved canvas + mask to %s", OUTPUT_DIR)

    # ---------------------------------------------------------------
    # Step 3: Set scheduler time_shift
    # ---------------------------------------------------------------
    step("Step 3: Configuring scheduler")
    time_shift = 3.0
    pipeline.scheduler = type(pipeline.scheduler).from_config(
        pipeline.scheduler.config, shift=time_shift,
    )
    log.info("Scheduler shift = %.1f", time_shift)

    # ---------------------------------------------------------------
    # Step 4: Run inpaint pipeline
    # ---------------------------------------------------------------
    step("Step 4: Running inpaint pipeline (outpaint mode)")
    seed = 42
    num_steps = 4  # Minimal steps for speed
    prompt = "a beautiful landscape extending to the right, vibrant colors"

    log.info("Params: %dx%d, seed=%d, steps=%d, control_scale=0.9", new_w, new_h, seed, num_steps)

    t_gen = time.time()
    try:
        result = pipeline(
            prompt=prompt,
            negative_prompt=None,
            height=new_h,
            width=new_w,
            image=canvas,
            mask_image=mask,
            control_context_scale=0.9,
            num_inference_steps=num_steps,
            guidance_scale=0.5,
            cfg_truncation=0.9,
            max_sequence_length=512,
            generator=torch.Generator(DEVICE).manual_seed(seed),
        )
    except Exception as e:
        log.error("Pipeline FAILED: %s", e)
        log.error(traceback.format_exc())
        return 1

    elapsed = time.time() - t_gen
    log.info("Pipeline completed in %.1fs", elapsed)

    # ---------------------------------------------------------------
    # Step 5: Validate output
    # ---------------------------------------------------------------
    step("Step 5: Validating output")

    images = result.images
    log.info("Got %d image(s)", len(images))

    img = images[0]
    arr = np.array(img)
    log.info("Output shape: %s, dtype: %s", arr.shape, arr.dtype)
    log.info("Output min=%d, max=%d, mean=%.1f", arr.min(), arr.max(), arr.mean())

    output_path = str(OUTPUT_DIR / "test_outpaint_result.png")
    img.save(output_path)
    log.info("Saved result to %s", output_path)

    # Basic sanity checks
    assert arr.max() > 10, f"Output too dark (max={arr.max()})"
    assert arr.shape[0] > 0 and arr.shape[1] > 0, "Output has zero dimensions"
    assert len(arr.shape) == 3 and arr.shape[2] == 3, f"Expected RGB, got shape {arr.shape}"

    step("ALL PASSED — Outpaint E2E test complete")
    log.info("  Output: %s", output_path)
    log.info("  Time: pipeline=%.1fs, total=%.1fs", elapsed, time.time() - t0)

    # Cleanup
    del pipeline, transformer, vae, text_encoder
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
