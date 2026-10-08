#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

mkdir -p "${RUNTIME_ROOT}"
python3.12 -m venv "${RUNTIME_VENV_DIR}"
"${RUNTIME_VENV_DIR}/bin/pip" install --upgrade pip
"${RUNTIME_VENV_DIR}/bin/pip" install 'litert-lm==0.16.1'

site_packages="$("${RUNTIME_VENV_DIR}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
server_patch_file="${PROJECT_DIR}/patches/litert-lm-openai-optional-constrained-decoding.patch"
if patch --batch --dry-run --silent -R -p2 -d "${site_packages}" < "${server_patch_file}"; then
  : # Already applied.
elif patch --batch --dry-run --silent -p2 -d "${site_packages}" < "${server_patch_file}"; then
  patch --batch --silent -p2 -d "${site_packages}" < "${server_patch_file}"
else
  echo "Installed LiteRT-LM server patch cannot be applied cleanly." >&2
  exit 1
fi

"${RUNTIME_VENV_DIR}/bin/litert-lm" --version
