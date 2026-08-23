#!/usr/bin/env python3
"""Download selected model files and link them into the headless runtime."""

from pathlib import Path

from huggingface_hub import hf_hub_download


COMFY_MODELS = Path("/opt/ComfyUI/models")

FILES = (
    (
        "black-forest-labs/FLUX.2-klein-4b-nvfp4",
        "flux-2-klein-4b-nvfp4.safetensors",
        COMFY_MODELS / "diffusion_models" / "flux-2-klein-4b-nvfp4.safetensors",
    ),
    (
        "ponpoke/flux2-klein-4b-uncensored-text-encoder",
        "flux2-klein-4b-uncensored-text-encoder/model.safetensors",
        COMFY_MODELS / "text_encoders" / "qwen_3_4b_uncensored.safetensors",
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


if __name__ == "__main__":
    main()
