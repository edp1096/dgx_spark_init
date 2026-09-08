#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output=${1:-runs/samples}
mkdir -p "$output"
output=$(cd "$output" && pwd)
image_id=$(docker inspect "${DOCUMENT_CONTAINER:-sparktalk-extra-documents}" --format '{{.Image}}')
source_tag="sparktalk-documents-validation-source:${image_id#sha256:}"
docker tag "$image_id" "$source_tag"
docker build --build-arg "DOCUMENT_IMAGE=$source_tag" -f validation/Dockerfile.office -t sparktalk-documents-validation .
docker run --rm --init --network host --user "$(id -u):$(id -g)" --memory=1g --cpus=2 --pids-limit=128 --read-only --tmpfs /tmp:rw,nosuid,size=512m --cap-drop ALL \
  -e "DOCUMENT_IMAGE_ID=$image_id" -e "DOCUMENTS_ENDPOINT=${DOCUMENTS_ENDPOINT:-http://127.0.0.1:8696}" \
  -v "$PWD/validation:/validation:ro" -v "$output:/out" \
  sparktalk-documents-validation node /validation/office-roundtrip.mjs /out
