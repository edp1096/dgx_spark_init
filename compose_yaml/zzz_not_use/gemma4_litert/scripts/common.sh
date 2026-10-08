#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${GEMMA4_LITERT_BUILD_ROOT:-${PROJECT_DIR}}"
BUILDER_DIR="${BUILD_ROOT}/builder"
ARTIFACT_DIR="${BUILD_ROOT}/artifacts"
VENV_DIR="${BUILDER_DIR}/venv"
RUNTIME_ROOT="${GEMMA4_LITERT_RUNTIME_ROOT:-${HOME}/.local/share/gemma4-litert}"
RUNTIME_VENV_DIR="${RUNTIME_ROOT}/venv"
SOURCE_MODEL_DIR="${BUILD_ROOT}/models/source-huihui-gemma4-e2b"
OUTPUT_DIR="${BUILD_ROOT}/output/Huihui-gemma-4-E2B-it-abliterated-litert-lm"

LITERT_REF="d4d1ea30bdcf1f018a019bc8229a566323cd6388"
LITERT_TORCH_REF="a6f8ea45be7d16d5db7363e404a846a94c1061a8"
LITERT_LM_REF="00ba5342028ec03c165b7adf1395e9add2640a04"
MODEL_REF="3d1e3d50d7a04585ce4ded197b2fd7a90c04647c"

clone_at_ref() {
  local repository="$1"
  local ref="$2"
  local destination="$3"

  if [[ -d "${destination}/.git" ]]; then
    return
  fi
  mkdir -p "${destination}"
  git -C "${destination}" init
  git -C "${destination}" remote add origin "${repository}"
  git -C "${destination}" fetch --depth=1 origin "${ref}"
  git -C "${destination}" checkout --detach FETCH_HEAD
}
