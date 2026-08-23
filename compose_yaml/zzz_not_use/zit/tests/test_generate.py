"""Z-Image Generate E2E tests — text-to-image with FP8/BF16 + base transformer.

Tests:
  1. PipelineManager loads ZIT pipeline (FP8)
  2. T2I generation produces valid image
  3. Base transformer inline denoising flow

Requires GPU + model weights.

Usage:
    cd /root/zit-ui && python tests/test_generate.py
    cd /root/zit-ui && python -m pytest tests/test_generate.py -v -s
"""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "ui"))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = str(Path.home() / ".cache" / "huggingface" / "hub" / "zit")
DEVICE = "cuda"
OUTPUT_DIR = Path("/tmp/zit-test-generate")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT = "Young Korean woman in red Hanbok, intricate embroidery, soft-lit outdoor background"
SEED = 42
HEIGHT = 1024
WIDTH = 768


def gpu_cleanup():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===========================================================================
# Test 1: PipelineManager loads successfully
# ===========================================================================
def test_pipeline_manager_loads():
    """PipelineManager.load_zit() should load without crash."""
    from pipeline_manager import PipelineManager
    mgr = PipelineManager()
    mgr.load_zit(use_fp8=True)

    assert mgr.zit_components is not None
    assert "pipeline" in mgr.zit_components or "transformer" in mgr.zit_components
    log.info("  PASS: PipelineManager loaded (FP8)")

    mgr.cleanup_all()
    del mgr
    gpu_cleanup()


# ===========================================================================
# Test 2: T2I generation produces valid image (via pipeline)
# ===========================================================================
def test_t2i_via_pipeline():
    """Generate image via PipelineManager pipeline, verify output."""
    from pipeline_manager import PipelineManager
    from diffusers import FlowMatchEulerDiscreteScheduler

    mgr = PipelineManager()
    mgr.load_zit(use_fp8=True)

    pipeline = mgr.zit_components["pipeline"]
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipeline.scheduler.config, shift=3.0)

    gen = torch.Generator(mgr.device).manual_seed(SEED)
    t0 = time.time()
    result = pipeline(
        prompt=PROMPT, height=HEIGHT, width=WIDTH,
        num_inference_steps=8, guidance_scale=0.5,
        cfg_normalization=False, cfg_truncation=0.9,
        max_sequence_length=512, generator=gen)
    elapsed = time.time() - t0

    img = result.images[0]
    arr = np.array(img)
    assert arr.ndim == 3 and arr.shape[2] == 3, f"Expected RGB, got {arr.shape}"
    assert arr.max() > 10, f"Image too dark (max={arr.max()})"

    out_path = str(OUTPUT_DIR / "t2i_fp8.png")
    img.save(out_path)
    log.info(f"  PASS: T2I generated in {elapsed:.1f}s → {out_path}")

    mgr.cleanup_all()
    del mgr
    gpu_cleanup()


# ===========================================================================
# Test 3: Base transformer inline denoising
# ===========================================================================
def test_base_transformer_denoising():
    """Load base transformer and run inline denoising loop (no pipeline wrapper)."""
    from pipeline_manager import PipelineManager
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.image_processor import VaeImageProcessor

    mgr = PipelineManager()
    mgr.load_zit(use_fp8=True)

    transformer = mgr.zit_components["transformer"]
    vae = mgr.zit_components["vae"]
    text_encoder = mgr.zit_components["text_encoder"]
    tokenizer = mgr.zit_components["tokenizer"]
    scheduler_config = mgr.zit_components["scheduler"].config

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config, shift=3.0)
    generator = torch.Generator(mgr.device).manual_seed(SEED)

    # Encode prompt
    messages = [{"role": "user", "content": PROMPT}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
    text_inputs = tokenizer([fmt], padding="max_length", max_length=512,
                            truncation=True, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(mgr.device)
    masks = text_inputs.attention_mask.to(mgr.device).bool()
    embeds = text_encoder(input_ids=input_ids, attention_mask=masks,
                          output_hidden_states=True).hidden_states[-2]
    prompt_embeds_list = [embeds[i][masks[i]] for i in range(len(embeds))]

    # Latents
    vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_scale_total = vae_scale * 2
    h_latent = 2 * (HEIGHT // vae_scale_total)
    w_latent = 2 * (WIDTH // vae_scale_total)
    latents = torch.randn((1, transformer.in_channels, h_latent, w_latent),
                          generator=generator, device=mgr.device, dtype=torch.float32)

    # Timesteps
    image_seq_len = (h_latent // 2) * (w_latent // 2)
    def _calc_shift(seq_len, base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15):
        m = (max_shift - base_shift) / (max_seq - base_seq)
        return seq_len * m + (base_shift - m * base_seq)

    mu = _calc_shift(image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15))
    scheduler.sigma_min = 0.0
    scheduler.set_timesteps(8, device=mgr.device, mu=mu)

    t0 = time.time()
    for i, t in enumerate(scheduler.timesteps):
        if t == 0 and i == len(scheduler.timesteps) - 1:
            continue
        timestep = t.expand(latents.shape[0])
        timestep = (1000 - timestep) / 1000
        latent_input = latents.to(next(transformer.parameters()).dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))
        model_out = transformer(latent_input_list, timestep, prompt_embeds_list)[0]
        noise_pred = model_out.float() if not isinstance(model_out, list) else torch.stack([x.float() for x in model_out], dim=0)
        noise_pred = -noise_pred.squeeze(2)
        latents = scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

    # VAE decode
    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents_dec = (latents.to(vae.dtype) / vae.config.scaling_factor) + shift_factor
    image = vae.decode(latents_dec, return_dict=False)[0]
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale * 2)
    images = image_processor.postprocess(image, output_type="pil")
    elapsed = time.time() - t0

    arr = np.array(images[0])
    assert arr.max() > 10, f"Image too dark (max={arr.max()})"

    out_path = str(OUTPUT_DIR / "t2i_inline_denoising.png")
    images[0].save(out_path)
    log.info(f"  PASS: Inline denoising in {elapsed:.1f}s → {out_path}")

    mgr.cleanup_all()
    del mgr
    gpu_cleanup()


# ===========================================================================
# Runner
# ===========================================================================
def run_all():
    tests = [
        test_pipeline_manager_loads,
        test_t2i_via_pipeline,
        test_base_transformer_denoising,
    ]

    total = passed = failed = 0
    errors = []

    for fn in tests:
        total += 1
        name = fn.__name__
        log.info(f"\n[{total}] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            import traceback
            traceback.print_exc()
            log.error(f"  FAIL: {e}")

    log.info(f"\n{'=' * 60}")
    log.info(f"  Results: {passed}/{total} passed, {failed} failed")
    log.info(f"{'=' * 60}")
    if errors:
        for name, err in errors:
            log.error(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
