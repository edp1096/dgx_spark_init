#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

bundle="$(find "${OUTPUT_DIR}" -maxdepth 2 -name '*.litertlm' -print -quit)"
if [[ -z "${bundle}" ]]; then
  echo "No .litertlm bundle found in ${OUTPUT_DIR}." >&2
  exit 1
fi

echo "Bundle: ${bundle}"
ls -lh "${bundle}"

unpack_dir="$(mktemp -d)"
trap 'rm -rf -- "${unpack_dir}"' EXIT
"${VENV_DIR}/bin/litert-lm" unpack "${bundle}" --output-dir "${unpack_dir}" --allow-overwrite

required_files=(
  Section3_TFLiteModel_tf_lite_prefill_decode.tflite
  Section4_TFLiteModel_tf_lite_embedder.tflite
  Section5_TFLiteModel_tf_lite_vision_encoder.tflite
  Section6_TFLiteModel_tf_lite_vision_adapter.tflite
  Section7_TFLiteModel_tf_lite_per_layer_embedder.tflite
  LlmMetadataProto.pbtext
)
for filename in "${required_files[@]}"; do
  if [[ ! -f "${unpack_dir}/${filename}" ]]; then
    echo "Required multimodal component is missing: ${filename}" >&2
    exit 1
  fi
done

find "${unpack_dir}" -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
