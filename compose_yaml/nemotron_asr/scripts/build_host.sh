#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ref="${NEMO_SPEECH_REF:-97a15afa5caa9bce5baaa86c1184103877af4101}"
destination="$root/artifacts/$ref"

if [[ -e "$destination" ]]; then
  echo "refusing to overwrite existing artifact: $destination" >&2
  exit 1
fi

mkdir -p "$destination"
docker build \
  --target host-artifacts \
  --build-arg "NEMO_SPEECH_REF=$ref" \
  --build-arg "CUDA_ARCH=${NEMO_CUDA_ARCH:-121}" \
  --build-arg "BUILD_JOBS=${NEMO_BUILD_JOBS:-2}" \
  --output "type=local,dest=$destination" \
  "$root"
ln -sfn "$ref" "$root/artifacts/current"
echo "exported: $destination"
