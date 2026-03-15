"""FP8 transformer NaN/Inf verification — hooks every module output.

Usage:
    python tests/test_fp8_nan_check.py          # Z-Image Turbo (default)
    python tests/test_fp8_nan_check.py --model zib
    python tests/test_fp8_nan_check.py --model klein
    python tests/test_fp8_nan_check.py --model klein_base
"""
import sys
sys.path.insert(0, "Z-Image/src")
sys.path.insert(0, "flux2/src")
sys.path.insert(0, "app/ui")

import argparse
import os
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = "/root/.cache/huggingface/hub/zifk"
DEVICE = "cuda"


def make_nan_hook(name, nan_found):
    def hook(module, input, output):
        if len(nan_found) > 10:
            return
        outputs = [output] if isinstance(output, torch.Tensor) else (output if isinstance(output, tuple) else [])
        for i, o in enumerate(outputs):
            if isinstance(o, torch.Tensor) and o.is_floating_point():
                try:
                    nan_c = torch.isnan(o).sum().item()
                    inf_c = torch.isinf(o).sum().item()
                except Exception:
                    continue
                if nan_c > 0 or inf_c > 0:
                    nan_found.append(f"{name}: nan={nan_c} inf={inf_c}")
                    log.info(f"  !!! {name}: nan={nan_c} inf={inf_c}")
    return hook


def check_zimage(model_type="turbo"):
    from utils import load_from_local_dir
    from safetensors.torch import load_file
    from pipeline_manager import _patch_transformer_q8, _load_fp8_weight_scales

    is_turbo = model_type == "turbo"
    subdir = "Z-Image-Turbo" if is_turbo else "Z-Image"
    label = "Z-Image Turbo" if is_turbo else "Z-Image Base"

    log.info(f"Loading {label}...")
    components = load_from_local_dir(
        f"{MODEL_DIR}/{subdir}", device=DEVICE, dtype=torch.bfloat16,
        verbose=False, compile=False,
    )
    transformer = components["transformer"]

    # Apply FP8 patch for Turbo only
    if is_turbo:
        fp8_file = f"{MODEL_DIR}/{subdir}/transformer/model_fp8.safetensors"
        fp8_state = load_file(fp8_file, device=DEVICE)
        transformer.load_state_dict(fp8_state, strict=False, assign=True)
        del fp8_state
        torch.cuda.empty_cache()
        weight_scales = _load_fp8_weight_scales(fp8_file)
        _patch_transformer_q8(transformer, weight_scales=weight_scales)
        transformer.dtype = torch.bfloat16

    nan_found = []
    hooks = []
    for name, module in transformer.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_nan_hook(name, nan_found)))

    steps = 4 if is_turbo else 28
    cfg = 0.0 if is_turbo else 3.5
    from zimage import generate
    generate(**components, prompt="test", height=1024, width=1024,
             num_inference_steps=steps, guidance_scale=cfg,
             generator=torch.Generator(DEVICE).manual_seed(42))

    for h in hooks:
        h.remove()
    return nan_found


def check_klein(variant="distilled"):
    from zifk_config import (KLEIN_MODEL_FILE, KLEIN_BASE_MODEL_FILE,
                              KLEIN_AE_FILE, KLEIN_DISTILLED, KLEIN_BASE)
    from pipeline_manager import _Q8_AVAILABLE, _patch_transformer_q8, _load_fp8_weight_scales

    is_distilled = variant == "distilled"
    model_file = KLEIN_MODEL_FILE if is_distilled else KLEIN_BASE_MODEL_FILE
    klein_variant = KLEIN_DISTILLED if is_distilled else KLEIN_BASE
    env_key = "KLEIN_4B_MODEL_PATH" if is_distilled else "KLEIN_4B_BASE_MODEL_PATH"
    os.environ[env_key] = f"{MODEL_DIR}/{model_file}"
    os.environ["AE_MODEL_PATH"] = f"{MODEL_DIR}/{KLEIN_AE_FILE}"

    from flux2.util import load_flow_model, load_text_encoder, load_ae
    from flux2.sampling import batched_prc_img, batched_prc_txt, denoise, denoise_cfg, get_schedule, scatter_ids

    log.info(f"Loading Klein {'Distilled' if is_distilled else 'Base'}...")
    model = load_flow_model(klein_variant, device=DEVICE)
    model.eval()

    if any(p.dtype == torch.float8_e4m3fn for p in model.parameters()) and _Q8_AVAILABLE:
        weight_scales = _load_fp8_weight_scales(f"{MODEL_DIR}/{model_file}")
        _patch_transformer_q8(model, weight_scales=weight_scales)
        model.dtype = torch.bfloat16

    nan_found = []
    hooks = []
    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_nan_hook(name, nan_found)))

    text_encoder = load_text_encoder(klein_variant, device=DEVICE)
    text_encoder.eval()
    ae = load_ae(klein_variant)
    ae.eval()

    with torch.no_grad():
        if is_distilled:
            ctx = text_encoder(["test"]).to(torch.bfloat16)
        else:
            ctx = torch.cat([text_encoder([""]).to(torch.bfloat16),
                             text_encoder(["test"]).to(torch.bfloat16)], dim=0)
        ctx, ctx_ids = batched_prc_txt(ctx)
        x = torch.randn(1, 128, 64, 64, dtype=torch.bfloat16, device=DEVICE,
                         generator=torch.Generator(DEVICE).manual_seed(42))
        x, x_ids = batched_prc_img(x)
        steps = 4 if is_distilled else 28
        timesteps = get_schedule(steps, x.shape[1])
        if is_distilled:
            denoise(model, x, x_ids, ctx, ctx_ids, timesteps=timesteps, guidance=1.0)
        else:
            denoise_cfg(model, x, x_ids, ctx, ctx_ids, timesteps=timesteps, guidance=4.0)

    for h in hooks:
        h.remove()
    return nan_found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="zit", choices=["zit", "zib", "klein", "klein_base", "all"])
    args = parser.parse_args()

    targets = ["zit", "zib", "klein", "klein_base"] if args.model == "all" else [args.model]

    for target in targets:
        if target == "zit":
            nans = check_zimage("turbo")
        elif target == "zib":
            nans = check_zimage("base")
        elif target == "klein":
            nans = check_klein("distilled")
        elif target == "klein_base":
            nans = check_klein("base")

        label = {"zit": "Z-Image Turbo", "zib": "Z-Image Base",
                 "klein": "Klein Distilled", "klein_base": "Klein Base"}[target]
        if nans:
            log.error(f"FAIL — {label}: {len(nans)} NaN/Inf locations")
            for n in nans[:5]:
                log.error(f"  {n}")
        else:
            log.info(f"PASS — {label}: 0 NaN/Inf")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
