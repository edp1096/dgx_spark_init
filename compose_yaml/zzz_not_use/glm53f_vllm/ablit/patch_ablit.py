#!/usr/bin/env python3
"""Install the GLM-5.3 load-time o_proj hook into a running container."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

SITE = Path(os.environ.get("VLLM_SITE", "/usr/local/lib/python3.12/dist-packages/vllm"))
MODEL_DIR = SITE / "models/glm5next/nvidia"
MODEL_FILE = MODEL_DIR / "model.py"
MTP_FILE = MODEL_DIR / "mtp.py"
RUNTIME_SOURCE = Path(__file__).with_name("ablit_runtime.py")
RUNTIME_TARGET = MODEL_DIR / "glm53_ablit.py"
MARKER = "GLM53-ABLIT-HOOK"
HOOK = (
    "        from vllm.models.glm5next.nvidia.glm53_ablit import maybe_apply "
    "as _glm53_ablit_apply  # GLM53-ABLIT-HOOK\n"
    "        _glm53_ablit_apply(self)  # GLM53-ABLIT-HOOK\n"
)
MODEL_TAIL = """                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        return loaded_params"""
MTP_TAIL = """                    f"missing from checkpoint."
                )
        return loaded_params"""


def patch(path: Path, target: str) -> None:
    source = path.read_text()
    if MARKER in source:
        return
    if source.count(target) != 1:
        raise SystemExit(f"{path}: Entrpi image changed; expected one hook target")
    replacement = target.replace(
        "        return loaded_params", HOOK + "        return loaded_params"
    )
    path.write_text(source.replace(target, replacement))


def main() -> None:
    for path in (MODEL_FILE, MTP_FILE, RUNTIME_SOURCE):
        if not path.is_file():
            raise SystemExit(f"required ablit file is missing: {path}")
    shutil.copyfile(RUNTIME_SOURCE, RUNTIME_TARGET)
    patch(MODEL_FILE, MODEL_TAIL)
    patch(MTP_FILE, MTP_TAIL)
    print("GLM-5.3 abliteration load hook installed")


if __name__ == "__main__":
    main()
