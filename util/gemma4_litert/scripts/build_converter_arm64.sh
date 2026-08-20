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
if [[ ! -x "${BUILDER_DIR}/bin/bazelisk" ]]; then
  GOBIN="${BUILDER_DIR}/bin" go install github.com/bazelbuild/bazelisk@latest
fi
ln -sfn "${BUILDER_DIR}/bin/bazelisk" "${BUILDER_DIR}/bin/bazel"

cd "${BUILDER_DIR}/LiteRT"
export PATH="${BUILDER_DIR}/bin:${PATH}"
export HERMETIC_PYTHON_VERSION=3.12
export PYTHON_BIN_PATH=/usr/bin/python3.12
export PYTHON_LIB_PATH=/usr/local/lib/python3.12/dist-packages
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_SET_ANDROID_WORKSPACE=0
export CC_OPT_FLAGS='-Wno-sign-compare -Wno-c++20-designator -Wno-gnu-inline-cpp-without-extern'

if [[ ! -f .litert_configure.bazelrc ]]; then
  yes '' | python3 configure.py
fi

bazel build -c opt \
  --cxxopt=-std=gnu++17 \
  --copt=-O3 \
  --repo_env=USE_PYWRAP_RULES=True \
  --action_env=HERMETIC_PYTHON_VERSION=3.12 \
  --config=litert_converter \
  --jobs="${BUILD_JOBS:-10}" \
  //ci/tools/python/wheel:litert_converter_wheel

cp -f bazel-bin/ci/tools/python/wheel/dist/litert_converter-*-aarch64.whl \
  "${ARTIFACT_DIR}/wheels/"
