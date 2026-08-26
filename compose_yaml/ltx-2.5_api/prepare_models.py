import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError


BASE_REPO = "Lightricks/LTX-2.5"
BASE_FILES = (
    "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
)

A2V_FILES = (
    "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
)

PUBLIC_FILES = (
    (
        "Muapi/ltx-2.3-ltx2-better-nsfw-motion",
        "ltx-2.3-ltx2-better-nsfw-motion.safetensors",
        "loras/ltx-2.3-ltx2-better-nsfw-motion.safetensors",
    ),
)


def required_paths(root: Path) -> list[Path]:
    paths = [root / filename for filename in BASE_FILES]
    paths.extend(root / relative_target for _, _, relative_target in PUBLIC_FILES)
    return paths


def missing_paths(root: Path) -> list[Path]:
    return [path for path in required_paths(root) if not path.is_file()]


def a2v_missing_paths(root: Path) -> list[Path]:
    return [root / filename for filename in A2V_FILES if not (root / filename).is_file()]


def download(
    repo: str,
    filename: str,
    local_dir: Path,
    target: Path,
    token: str | None,
) -> None:
    if target.is_file():
        print(f"ready: {target}", flush=True)
        return
    print(f"downloading: {repo}/{filename}", flush=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=local_dir,
        token=token,
    )


def prepare_all(root: Path, token: str | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Public optional assets can always be prepared, even before the user has
    # configured access to the gated official checkpoint.
    for repo, filename, relative_target in PUBLIC_FILES:
        target = root / relative_target
        download(repo, filename, target.parent, target, token)

    try:
        for filename in BASE_FILES:
            download(BASE_REPO, filename, root, root / filename, token)
        for filename in A2V_FILES:
            download(BASE_REPO, filename, root, root / filename, token)
    except GatedRepoError as exc:
        raise SystemExit(
            "LTX-2.5 download requires an approved Hugging Face account and HF_TOKEN. "
            "Accept the license at https://huggingface.co/Lightricks/LTX-2.5, "
            "then enter its read token in Spark Media settings."
        ) from exc


def main() -> None:
    root = Path(os.environ.get("LTX_MODEL_DIR", "/models/ltx-2.5"))
    prepare_all(root, os.environ.get("HF_TOKEN") or None)


if __name__ == "__main__":
    main()
