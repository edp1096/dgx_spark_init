"""FP8 tests — weight analysis + NaN/Inf verification during inference.

Tests:
  1. FP8 weight file structure (dtype distribution, scales)
  2. NaN/Inf check via forward hooks during generation

Requires GPU + model weights.

Usage:
    cd /root/zit-ui && python tests/test_fp8.py
    cd /root/zit-ui && python tests/test_fp8.py --skip-inference
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "ui"))

import torch
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "zit"
DEVICE = "cuda"


# ===========================================================================
# Test 1: FP8 weight file analysis
# ===========================================================================
def test_fp8_weight_info():
    """Analyze FP8 safetensors: dtype distribution, scale keys, absmax range."""
    fp8_path = MODEL_DIR / "Z-Image-Turbo" / "transformer" / "model_fp8.safetensors"
    if not fp8_path.exists():
        log.warning(f"  SKIP: {fp8_path} not found")
        return

    sd = load_file(str(fp8_path), device="cpu")

    # Dtype distribution
    dtypes = {}
    for k, t in sd.items():
        d = str(t.dtype)
        dtypes[d] = dtypes.get(d, 0) + 1

    log.info(f"  Keys: {len(sd)}")
    for d, c in sorted(dtypes.items()):
        log.info(f"    {d}: {c}")

    fp8_keys = [k for k in sd if sd[k].dtype == torch.float8_e4m3fn]
    scale_keys = [k for k in sd if "scale" in k.lower()]

    assert len(fp8_keys) > 0, "Should have FP8 weight keys"
    log.info(f"  FP8 weights: {len(fp8_keys)}, Scale keys: {len(scale_keys)}")

    # Check absmax range
    if fp8_keys:
        all_absmax = [sd[k].float().abs().max().item() for k in fp8_keys]
        log.info(f"  FP8 absmax range: [{min(all_absmax):.1f}, {max(all_absmax):.1f}]")
        assert max(all_absmax) < 500, f"FP8 absmax too large: {max(all_absmax)}"

    # Check scale values
    if scale_keys:
        for k in scale_keys[:3]:
            t = sd[k]
            val = t.float().item() if t.numel() == 1 else f"shape={tuple(t.shape)}"
            log.info(f"    {k}: {val}")

    del sd
    log.info("  PASS: FP8 weight analysis")


# ===========================================================================
# Test 2: NaN/Inf verification during inference
# ===========================================================================
def test_fp8_nan_check():
    """Hook every module output during generation to detect NaN/Inf."""
    from pipeline_manager import PipelineManager, _patch_transformer_q8, _load_fp8_weight_scales

    mgr = PipelineManager()
    mgr.load_zit(use_fp8=True)
    transformer = mgr.zit_components["transformer"]

    nan_found = []

    def make_hook(name):
        def hook(module, input, output):
            if len(nan_found) > 10:
                return
            outputs = [output] if isinstance(output, torch.Tensor) else (output if isinstance(output, tuple) else [])
            for o in outputs:
                if isinstance(o, torch.Tensor) and o.is_floating_point():
                    try:
                        nan_c = torch.isnan(o).sum().item()
                        inf_c = torch.isinf(o).sum().item()
                    except Exception:
                        continue
                    if nan_c > 0 or inf_c > 0:
                        nan_found.append(f"{name}: nan={nan_c} inf={inf_c}")
                        log.warning(f"  !!! {name}: nan={nan_c} inf={inf_c}")
        return hook

    hooks = []
    for name, module in transformer.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run inference
    from diffusers import FlowMatchEulerDiscreteScheduler
    pipeline = mgr.zit_components["pipeline"]
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipeline.scheduler.config, shift=3.0)
    gen = torch.Generator(mgr.device).manual_seed(42)
    pipeline(prompt="test portrait", height=1024, width=1024,
             num_inference_steps=4, guidance_scale=0.5,
             cfg_normalization=False, cfg_truncation=0.9,
             max_sequence_length=512, generator=gen)

    for h in hooks:
        h.remove()

    if nan_found:
        log.error(f"  FAIL: {len(nan_found)} NaN/Inf locations")
        for n in nan_found[:5]:
            log.error(f"    {n}")
        raise AssertionError(f"FP8 NaN/Inf detected: {len(nan_found)} locations")

    log.info("  PASS: 0 NaN/Inf during FP8 inference")

    mgr.cleanup_all()
    del mgr
    torch.cuda.empty_cache()


# ===========================================================================
# Runner
# ===========================================================================
def run_all(skip_inference=False):
    tests = [test_fp8_weight_info]
    if not skip_inference:
        tests.append(test_fp8_nan_check)

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

    log.info(f"\n{'=' * 60}")
    log.info(f"  Results: {passed}/{total} passed, {failed} failed")
    log.info(f"{'=' * 60}")
    if errors:
        for name, err in errors:
            log.error(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-inference", action="store_true",
                        help="Only run weight analysis, skip GPU inference")
    args = parser.parse_args()
    sys.exit(run_all(skip_inference=args.skip_inference))
