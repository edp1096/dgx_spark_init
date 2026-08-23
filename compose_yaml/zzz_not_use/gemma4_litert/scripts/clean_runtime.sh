#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

model_id="huihui-gemma4-e2b"
model_dir="${HOME}/.litert-lm/models/${model_id}"
runtime_cli="${RUNTIME_VENV_DIR}/bin/litert-lm"

case "${RUNTIME_ROOT}" in
  ""|/|"${HOME}")
    echo "Refusing to remove unsafe runtime path: ${RUNTIME_ROOT}" >&2
    exit 1
    ;;
esac

if [[ -d "${model_dir}" ]]; then
  if [[ -x "${runtime_cli}" ]]; then
    "${runtime_cli}" delete "${model_id}"
  else
    rm -rf -- "${model_dir}"
  fi
fi

if [[ -d "${RUNTIME_ROOT}" ]]; then
  rm -rf -- "${RUNTIME_ROOT}"
fi

echo "Removed ${model_id} and the project LiteRT-LM runtime. Other registered models were preserved."
