#!/usr/bin/env python3
"""Download the Krea 2 Turbo NVFP4 components used by ComfyUI."""

import hashlib
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download


COMFY_MODELS = Path("/opt/ComfyUI/models")

FILES = (
    (
        "Comfy-Org/Krea-2",
        "diffusion_models/krea2_turbo_nvfp4.safetensors",
        COMFY_MODELS / "diffusion_models" / "krea2_turbo_nvfp4.safetensors",
    ),
    (
        "Comfy-Org/Krea-2",
        "diffusion_models/krea2_turbo_int8_convrot.safetensors",
        COMFY_MODELS / "diffusion_models" / "krea2_turbo_int8_convrot.safetensors",
    ),
    (
        "Comfy-Org/Krea-2",
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        COMFY_MODELS / "text_encoders" / "qwen3vl_4b_fp8_scaled.safetensors",
    ),
    (
        "Comfy-Org/Krea-2",
        "text_encoders/qwen3vl_4b_bf16.safetensors",
        COMFY_MODELS / "text_encoders" / "qwen3vl_4b_bf16.safetensors",
    ),
    (
        "Comfy-Org/Krea-2",
        "vae/qwen_image_vae.safetensors",
        COMFY_MODELS / "vae" / "qwen_image_vae.safetensors",
    ),
    (
        "artsyww/KREA2REALVAE",
        "krea2RealVae_v10.safetensors",
        COMFY_MODELS / "vae" / "krea2RealVae_v10.safetensors",
    ),
    (
        "Patil/Krea-2-depth-controlnet",
        "depth-control-lora.safetensors",
        COMFY_MODELS / "loras" / "krea2-depth-control-lora.safetensors",
    ),
    (
        "conradlocke/krea2-identity-edit",
        "krea2_identity_edit_v1_2.safetensors",
        COMFY_MODELS / "loras" / "krea2_identity_edit_v1_2.safetensors",
    ),
    (
        "uzumix/krea2filterbypass3.safetensors",
        "krea2filterbypass3.safetensors",
        COMFY_MODELS / "loras" / "krea2filterbypass3.safetensors",
    ),
    (
        "reverentelusarca/krea2-detail-enhancer-edit-lora",
        "krea-detail-enhancer-exp.safetensors",
        COMFY_MODELS / "loras" / "krea-detail-enhancer-exp.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "split_files/vae/wan_2.1_vae.safetensors",
        COMFY_MODELS / "vae" / "wan_2.1_vae.safetensors",
    ),
    (
        "nynxz/NK2E",
        "comfy/v0.3/NK2E-v0.3.safetensors",
        COMFY_MODELS / "loras" / "NK2E-v0.3.safetensors",
    ),
    (
        "nynxz/NK2E",
        "comfy/canny_v0.1/NK2E-canny-v0.1.safetensors",
        COMFY_MODELS / "loras" / "NK2E-canny-v0.1.safetensors",
    ),
    (
        "yijunwang2/krea2-anypaint",
        "krea2_anypaint_rank32.safetensors",
        COMFY_MODELS / "loras" / "krea2_anypaint_rank32.safetensors",
    ),
    (
        "Comfy-Org/Krea-2",
        "loras/krea2_retroanime.safetensors",
        COMFY_MODELS / "loras" / "krea2_retroanime.safetensors",
    ),
    *(
        (
            "Comfy-Org/Krea-2",
            f"loras/krea2_{name}.safetensors",
            COMFY_MODELS / "loras" / f"krea2_{name}.safetensors",
        )
        for name in (
            "darkbrush",
            "dotmatrix",
            "kidsdrawing",
            "neondrip",
            "rainywindow",
            "softwatercolor",
            "sunsetblur",
            "vintagetarot",
            "style_reference",
        )
    ),
)

URL_FILES = (
    (
        "https://raw.githubusercontent.com/CliffNodes/fedor_bypass/024bf4cd96c824321807ce93574a57ca72867366/fedor_bypass.safetensors",
        "312024b593c0a0561b18be3f04bd7d92810cab05af28183dc3ced91c18913eb2",
        Path("/root/.cache/huggingface/media-models/fedor_bypass.safetensors"),
        COMFY_MODELS / "loras" / "fedor_bypass.safetensors",
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
    for url, expected_sha256, cached, destination in URL_FILES:
        cached.parent.mkdir(parents=True, exist_ok=True)
        if not cached.exists() or hashlib.sha256(cached.read_bytes()).hexdigest() != expected_sha256:
            with urllib.request.urlopen(url) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected_sha256:
                raise RuntimeError(f"checksum mismatch for {url}")
            cached.write_bytes(data)
        replace_symlink(destination, cached)
        print(f"MODEL {destination.name} -> {cached}", flush=True)


if __name__ == "__main__":
    main()
