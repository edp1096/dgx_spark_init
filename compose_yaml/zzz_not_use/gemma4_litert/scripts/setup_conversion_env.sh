#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

clone_at_ref https://github.com/google-ai-edge/litert-torch.git "${LITERT_TORCH_REF}" "${BUILDER_DIR}/litert-torch"
clone_at_ref https://github.com/google-ai-edge/LiteRT-LM.git "${LITERT_LM_REF}" "${BUILDER_DIR}/LiteRT-LM"

patch_file="${PROJECT_DIR}/patches/litert-torch-transformers5-heterogeneous-cache.patch"
if git -C "${BUILDER_DIR}/litert-torch" apply --check "${patch_file}" 2>/dev/null; then
  git -C "${BUILDER_DIR}/litert-torch" apply "${patch_file}"
elif ! git -C "${BUILDER_DIR}/litert-torch" apply --reverse --check "${patch_file}" 2>/dev/null; then
  echo "LiteRT Torch compatibility patch cannot be applied cleanly." >&2
  exit 1
fi

wheel_path="$(find "${ARTIFACT_DIR}/wheels" -maxdepth 1 -name 'litert_converter-*-aarch64.whl' -print -quit)"
if [[ -z "${wheel_path}" ]]; then
  echo "ARM64 litert-converter wheel is missing. Run make converter first." >&2
  exit 1
fi

python3.12 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install "${wheel_path}"
"${VENV_DIR}/bin/pip" install \
  --find-links "${ARTIFACT_DIR}/wheels" \
  -r "${BUILDER_DIR}/litert-torch/requirements.txt"
"${VENV_DIR}/bin/pip" install --no-deps -e "${BUILDER_DIR}/litert-torch"

site_packages="$("${VENV_DIR}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
server_patch_file="${PROJECT_DIR}/patches/litert-lm-openai-optional-constrained-decoding.patch"
if patch --batch --dry-run --silent -R -p2 -d "${site_packages}" < "${server_patch_file}"; then
  : # Already applied.
elif patch --batch --dry-run --silent -p2 -d "${site_packages}" < "${server_patch_file}"; then
  patch --batch --silent -p2 -d "${site_packages}" < "${server_patch_file}"
else
  echo "Installed LiteRT-LM server patch cannot be applied cleanly." >&2
  exit 1
fi

"${VENV_DIR}/bin/python" - <<'PY'
import platform
import torch
from litert_converter.mlir._mlir_libs import converter_api_ext

assert platform.machine() == "aarch64"
print("converter:", converter_api_ext.__file__)
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
