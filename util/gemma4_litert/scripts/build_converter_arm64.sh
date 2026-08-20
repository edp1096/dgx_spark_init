#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This target builds the native Linux ARM64 converter (host: $(uname -m))." >&2
  exit 1
fi

mkdir -p "${BUILDER_DIR}/bin" "${ARTIFACT_DIR}/wheels"
clone_at_ref https://github.com/google-ai-edge/LiteRT.git "${LITERT_REF}" "${BUILDER_DIR}/LiteRT"

if ! command -v go >/dev/null; then
  echo "Go is required to install Bazelisk." >&2
  exit 1
fi
BAZELISK_VERSION="${BAZELISK_VERSION:-v1.25.0}"
bazelisk_bin="${BUILDER_DIR}/bin/bazelisk-${BAZELISK_VERSION}"
if [[ ! -x "${bazelisk_bin}" ]]; then
  GOBIN="${BUILDER_DIR}/bin" go install "github.com/bazelbuild/bazelisk@${BAZELISK_VERSION}"
  mv "${BUILDER_DIR}/bin/bazelisk" "${bazelisk_bin}"
fi
ln -sfn "${bazelisk_bin}" "${BUILDER_DIR}/bin/bazel"

cd "${BUILDER_DIR}/LiteRT"
export PATH="${BUILDER_DIR}/bin:${PATH}"
export HERMETIC_PYTHON_VERSION=3.12
export PYTHON_BIN_PATH="$(command -v python3.12)"
export PYTHON_LIB_PATH="$(python3.12 -c 'import site; print(site.getsitepackages()[0])')"
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_SET_ANDROID_WORKSPACE=0
export CC_OPT_FLAGS='-Wno-sign-compare -Wno-c++20-designator -Wno-gnu-inline-cpp-without-extern'

if [[ ! -f .litert_configure.bazelrc ]]; then
  python3.12 configure.py <<< $'\n\n'
fi

bazel_output_root="${BAZEL_OUTPUT_ROOT:-${BUILD_ROOT}/cache/bazel}"
bazel --output_user_root="${bazel_output_root}" build -c opt \
  --cxxopt=-std=gnu++17 \
  --copt=-O3 \
  --repo_env=USE_PYWRAP_RULES=True \
  --action_env=HERMETIC_PYTHON_VERSION=3.12 \
  --config=litert_converter \
  --jobs="${BUILD_JOBS:-10}" \
  //ci/tools/python/wheel:litert_converter_wheel

bazel_bin_dir="$(bazel --output_user_root="${bazel_output_root}" info bazel-bin)"
cp -f "${bazel_bin_dir}"/ci/tools/python/wheel/dist/litert_converter-*aarch64.whl \
  "${ARTIFACT_DIR}/wheels/"
