#!/usr/bin/env python3
"""Assemble FLUX.2 Klein 4B with a replacement Safetensors text encoder."""

import os
import json
from pathlib import Path

from huggingface_hub import snapshot_download


OFFICIAL_REPO = os.environ.get(
    "OFFICIAL_MODEL_REPO", "black-forest-labs/FLUX.2-klein-4B"
)
TEXT_ENCODER_REPO = os.environ.get(
    "TEXT_ENCODER_REPO", "ponpoke/flux2-klein-4b-uncensored-text-encoder"
)
TEXT_ENCODER_SUBDIR = "flux2-klein-4b-uncensored-text-encoder"
OUTPUT_DIR = Path(
    os.environ.get(
        "ASSEMBLED_MODEL_DIR",
        "/root/.cache/huggingface/local/flux2-klein-4b-uncensored",
    )
)


def replace_with_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def main() -> None:
    official = Path(snapshot_download(OFFICIAL_REPO))
    replacement_snapshot = Path(
        snapshot_download(
            TEXT_ENCODER_REPO,
            allow_patterns=[f"{TEXT_ENCODER_SUBDIR}/*"],
        )
    )
    replacement = replacement_snapshot / TEXT_ENCODER_SUBDIR

    required = (
        replacement / "config.json",
        replacement / "model.safetensors",
        replacement / "tokenizer.json",
        replacement / "tokenizer_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"replacement text encoder is incomplete: {missing}")
    if list(replacement.rglob("*.gguf")):
        raise RuntimeError("GGUF files unexpectedly appeared in the selected download")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("model_index.json", "scheduler", "tokenizer", "transformer", "vae"):
        target = official / name
        if not target.exists():
            raise RuntimeError(f"official model component is missing: {target}")
        replace_with_symlink(OUTPUT_DIR / name, target)

    # The replacement checkpoint is stored as FP16. The official FLUX.2 Klein
    # transformer runs as BF16, so make Transformers cast the replacement while
    # loading; otherwise the first projection fails with Half/BFloat16 mismatch.
    text_encoder_output = OUTPUT_DIR / "text_encoder"
    if text_encoder_output.is_symlink():
        text_encoder_output.unlink()
    text_encoder_output.mkdir(exist_ok=True)
    for source in replacement.iterdir():
        destination = text_encoder_output / source.name
        if source.name == "config.json":
            if destination.is_symlink():
                destination.unlink()
            config = json.loads(source.read_text(encoding="utf-8"))
            config["dtype"] = "bfloat16"
            destination.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            replace_with_symlink(destination, source)

    marker = OUTPUT_DIR / "ASSEMBLED_FROM"
    marker.write_text(
        f"base={OFFICIAL_REPO}\ntext_encoder={TEXT_ENCODER_REPO}\n"
        f"format=safetensors\nload_dtype=bfloat16\n",
        encoding="utf-8",
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
