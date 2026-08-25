#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
llama_ref="${LLAMA_CPP_REF:-c060ca974c773c7c3d17fd1b66dc9d312bc292c0}"
artifact_name="${llama_ref//\//_}"
artifact_dir="${LLAMA_ARTIFACT_ROOT:-${script_dir}/artifacts}/${artifact_name}"
image_repo="${LLAMA_IMAGE_REPO:-llama.cpp-spark}"
image_tag="${LLAMA_IMAGE_TAG:-${llama_ref//\//-}}"

if [[ ! -x "${artifact_dir}/llama-server" ]]; then
  LLAMA_CPP_REF="${llama_ref}" "${script_dir}/scripts/build_host.sh"
fi

docker build \
  --build-arg "LLAMA_CPP_REF=${llama_ref}" \
  -t "${image_repo}:${image_tag}" \
  "${script_dir}"

if [[ "${LLAMA_PUSH:-false}" == "true" ]]; then
  docker push "${image_repo}:${image_tag}"
fi

echo "Host runtime: ${artifact_dir}"
echo "Docker image: ${image_repo}:${image_tag}"
