#!/usr/bin/env python3
"""Download the official Z-Image Turbo NVFP4 components used by ComfyUI."""

from pathlib import Path

from huggingface_hub import hf_hub_download


COMFY_MODELS = Path("/opt/ComfyUI/models")
REPO = "Comfy-Org/z_image_turbo"

FILES = (
    (
        "split_files/diffusion_models/z_image_turbo_nvfp4.safetensors",
        COMFY_MODELS / "diffusion_models" / "z_image_turbo_nvfp4.safetensors",
    ),
    (
        "split_files/text_encoders/qwen_3_4b_fp4_mixed.safetensors",
        COMFY_MODELS / "text_encoders" / "qwen_3_4b_fp4_mixed.safetensors",
    ),
    (
        "split_files/vae/ae.safetensors",
        COMFY_MODELS / "vae" / "ae.safetensors",
    ),
    (
        "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors",
        COMFY_MODELS
        / "model_patches"
        / "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors",
        "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
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
    for item in FILES:
        filename, destination, *repo_override = item
        downloaded = Path(
            hf_hub_download(repo_id=repo_override[0] if repo_override else REPO, filename=filename)
        )
        replace_symlink(destination, downloaded)
        print(f"MODEL {destination.name} -> {downloaded}", flush=True)


if __name__ == "__main__":
    main()
