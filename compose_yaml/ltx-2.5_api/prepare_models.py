import os
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = "Lightricks/LTX-2.5"
FILES = (
    "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
)


def main() -> None:
    root = Path(os.environ.get("LTX_MODEL_DIR", "/models/ltx-2.5"))
    root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    for filename in FILES:
        target = root / filename
        if target.is_file():
            print(f"ready: {target}", flush=True)
            continue
        print(f"downloading: {REPO}/{filename}", flush=True)
        hf_hub_download(
            repo_id=REPO,
            filename=filename,
            local_dir=root,
            token=token,
        )


if __name__ == "__main__":
    main()
