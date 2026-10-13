#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
llama_ref="${LLAMA_CPP_REF:-c060ca974c773c7c3d17fd1b66dc9d312bc292c0}"
artifact_root="${LLAMA_ARTIFACT_ROOT:-${project_dir}/artifacts}"
artifact_name="${llama_ref//\//_}"
artifact_dir="${artifact_root}/${artifact_name}"

mkdir -p "${artifact_root}"
if [[ -e "${artifact_dir}" ]]; then
  echo "Host artifact already exists: ${artifact_dir}" >&2
  echo "Set LLAMA_CPP_REF to another ref or move the existing directory first." >&2
  exit 1
fi

stage_dir="$(mktemp -d "${artifact_root}/.build.XXXXXX")"
cleanup() {
  if [[ -d "${stage_dir}" ]]; then
    rm -rf -- "${stage_dir}"
  fi
}
trap cleanup EXIT

docker build \
  --build-arg "LLAMA_CPP_REF=${llama_ref}" \
  --target host-artifacts \
  --output "type=local,dest=${stage_dir}" \
  "${project_dir}"

test -x "${stage_dir}/llama-server"
mv "${stage_dir}" "${artifact_dir}"
ln -sfn "${artifact_name}" "${artifact_root}/current"
trap - EXIT

echo "Host runtime: ${artifact_dir}"
echo "Current link: ${artifact_root}/current"
