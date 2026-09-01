#!/usr/bin/env python3
"""Download and link the model set used by the H3 character-sheet workflow."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download


COMFY_MODELS = Path("/opt/ComfyUI/models")

FILES = (
    (
        "Comfy-Org/MiniMax-H3",
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        COMFY_MODELS / "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    ),
    (
        "Comfy-Org/MiniMax-H3",
        "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        COMFY_MODELS / "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    ),
    (
        "Comfy-Org/MiniMax-H3",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        COMFY_MODELS / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    ),
    (
        "Comfy-Org/MiniMax-H3",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        COMFY_MODELS / "vae/minimax_h3_audio_vae_fp32.safetensors",
    ),
    (
        "Comfy-Org/MiniMax-H3",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        COMFY_MODELS / "vae/minimax_h3_video_vae_fp16.safetensors",
    ),
    (
        "Mamad8/MiniMax-H3-Image-VAE",
        "minimax_h3_t1_image_vae_step1597.safetensors",
        COMFY_MODELS / "vae/minimax_h3_t1_image_vae_step1597.safetensors",
    ),
)


def link_file(repo: str, filename: str, destination: Path) -> str:
    downloaded = Path(hf_hub_download(repo_id=repo, filename=filename))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == downloaded.resolve():
            return f"ready: {destination.name}"
        destination.unlink()
    elif destination.exists():
        if destination.resolve() == downloaded.resolve():
            return f"ready: {destination.name}"
        raise RuntimeError(f"refusing to replace non-symlink model: {destination}")
    destination.symlink_to(downloaded)
    return f"linked: {destination.name}"


def main() -> None:
    # Separate repositories and files can download concurrently. Existing blobs
    # are resolved from the shared cache without network traffic.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(link_file, *entry): entry for entry in FILES}
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
