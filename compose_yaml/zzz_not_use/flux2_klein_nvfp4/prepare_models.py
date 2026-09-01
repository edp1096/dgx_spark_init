#!/usr/bin/env python3
"""Download/link selected model files into the headless runtime."""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


COMFY_MODELS = Path("/opt/ComfyUI/models")
HF_HOME = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
TEXT_ENCODER_DESTINATION = COMFY_MODELS / "text_encoders" / "qwen_3_4b_flux2.safetensors"
LOCAL_UNCENSORED_NVFP4 = (
    HF_HOME
    / "local"
    / "flux2-klein-4b-uncensored-nvfp4"
    / "qwen_3_4b_uncensored_nvfp4_flux2.safetensors"
)

FILES = (
    (
        "black-forest-labs/FLUX.2-klein-4b-nvfp4",
        "flux-2-klein-4b-nvfp4.safetensors",
        COMFY_MODELS / "diffusion_models" / "flux-2-klein-4b-nvfp4.safetensors",
    ),
    (
        "Comfy-Org/flux2-klein-4B",
        "split_files/vae/flux2-vae.safetensors",
        COMFY_MODELS / "vae" / "flux2-vae.safetensors",
    ),
)

def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink model path: {link}")
    link.symlink_to(target)


def main() -> None:
    for repo, filename, destination in FILES:
        downloaded = Path(hf_hub_download(repo_id=repo, filename=filename))
        replace_symlink(destination, downloaded)
        print(f"MODEL {destination.name} -> {downloaded}", flush=True)

    if not LOCAL_UNCENSORED_NVFP4.is_file():
        raise RuntimeError(
            "uncensored NVFP4 text encoder is missing; run "
            "quantize_uncensored_text_encoder.py first"
        )
    replace_symlink(TEXT_ENCODER_DESTINATION, LOCAL_UNCENSORED_NVFP4)
    print(
        f"MODEL {TEXT_ENCODER_DESTINATION.name} (uncensored NVFP4) -> "
        f"{LOCAL_UNCENSORED_NVFP4}",
        flush=True,
    )


if __name__ == "__main__":
    main()
