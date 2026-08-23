#!/usr/bin/env bash

set -euo pipefail

data_dir="${GEMMA4_LITERT_CONTAINER_DATA_DIR:-${HOME}/.local/share/gemma4-litert-build}"
marker="${data_dir}/.gemma4-litert-build-root"

case "${data_dir}" in
  ""|/|"${HOME}"|"${HOME}/.local"|"${HOME}/.local/share")
    echo "Refusing to remove unsafe container build path: ${data_dir}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${marker}" ]]; then
  echo "Refusing to remove an unmarked container build path: ${data_dir}" >&2
  exit 1
fi

rm -rf -- "${data_dir}"
echo "Removed container build cache, wheel, source model, and converted output: ${data_dir}"
