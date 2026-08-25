#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "${script_dir}/.." && pwd)

target_repo=${TRT_EDGE_DFLASH_TARGET_REPO:-/home/edp1096/workspace/heretic_models/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4}
draft_repo=${TRT_EDGE_DFLASH_DRAFT_REPO:-/home/edp1096/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2}
workspace_dir=${TRT_EDGE_DFLASH_WORKSPACE_DIR:-${project_dir}/workspace}
max_input_len=${TRT_EDGE_MAX_INPUT_LEN:-32768}
max_kv_capacity=${TRT_EDGE_MAX_KV_CACHE_CAPACITY:-32768}
engine_subdir=${TRT_EDGE_ENGINE_SUBDIR:-dflash-ddtree-32k}
phase=${1:-all}

target_mount=${target_repo}
target_dir=${target_repo}
target_container_dir=/model
if [[ -d "${target_repo}/snapshots" && -f "${target_repo}/refs/main" ]]; then
  target_revision=$(tr -d '\r\n' <"${target_repo}/refs/main")
  target_dir="${target_repo}/snapshots/${target_revision}"
  target_container_dir="/model/snapshots/${target_revision}"
fi

draft_mount=${draft_repo}
draft_dir=${draft_repo}
draft_container_dir=/draft
if [[ -d "${draft_repo}/snapshots" && -f "${draft_repo}/refs/main" ]]; then
  draft_revision=$(tr -d '\r\n' <"${draft_repo}/refs/main")
  draft_dir="${draft_repo}/snapshots/${draft_revision}"
  draft_container_dir="/draft/snapshots/${draft_revision}"
fi

for required in \
  "${target_dir}/config.json" \
  "${target_dir}/model.safetensors.index.json" \
  "${draft_dir}/config.json" \
  "${draft_dir}/model.safetensors"; do
  if [[ ! -e "${required}" ]]; then
    echo "required checkpoint file is missing: ${required}" >&2
    exit 1
  fi
done

case "${phase}" in
  all|export|base-export|draft-export|build|base-build|draft-build|vision) ;;
  *)
    echo "usage: $0 [all|export|base-export|draft-export|build|base-build|draft-build|vision]" >&2
    exit 2
    ;;
esac

export TRT_EDGE_MODEL_DIR="${target_mount}"
export TRT_EDGE_DRAFT_DIR="${draft_mount}"
export TRT_EDGE_WORKSPACE_DIR="${workspace_dir}"
mkdir -p "${workspace_dir}"
cd "${project_dir}"

if [[ "${phase}" == all || "${phase}" == export || "${phase}" == base-export || "${phase}" == draft-export ]]; then
  docker compose --profile tools build exporter
fi
if [[ "${phase}" == all || "${phase}" == export || "${phase}" == base-export ]]; then
  docker compose --profile tools run --rm exporter \
    "${target_container_dir}" /workspace/onnx/dflash-tree-base \
    --dflash-tree-base --dflash-draft-dir "${draft_container_dir}"
fi
if [[ "${phase}" == all || "${phase}" == export || "${phase}" == draft-export ]]; then
  docker compose --profile tools run --rm exporter \
    "${target_container_dir}" /workspace/onnx/dflash-draft \
    --dflash-draft --dflash-draft-dir "${draft_container_dir}"
fi

if [[ "${phase}" == all || "${phase}" == build || "${phase}" == base-build || "${phase}" == draft-build ]]; then
  docker compose stop server
fi
if [[ "${phase}" == all || "${phase}" == build || "${phase}" == base-build ]]; then
  docker compose --profile tools run --rm runtime \
    ./build/examples/llm/llm_build \
    --onnxDir /workspace/onnx/dflash-tree-base/llm \
    --engineDir "/workspace/engines/${engine_subdir}" \
    --maxBatchSize 1 \
    --maxInputLen "${max_input_len}" \
    --maxKVCacheCapacity "${max_kv_capacity}" \
    --maxVerifyTreeSize 16 \
    --specBase
fi
if [[ "${phase}" == all || "${phase}" == build || "${phase}" == draft-build ]]; then
  docker compose --profile tools run --rm runtime \
    ./build/examples/llm/llm_build \
    --onnxDir /workspace/onnx/dflash-draft/dflash_draft \
    --engineDir "/workspace/engines/${engine_subdir}" \
    --maxBatchSize 1 \
    --maxInputLen "${max_input_len}" \
    --maxKVCacheCapacity "${max_kv_capacity}" \
    --maxDraftTreeSize 16 \
    --specDraft
fi

if [[ "${phase}" == vision ]]; then
  if [[ ! -f "${workspace_dir}/onnx/dflash-tree-base/visual/model.onnx" ]]; then
    echo "DFlash tree-base visual ONNX is missing; run '$0 base-export' first" >&2
    exit 1
  fi
  docker compose stop server
  docker compose --profile tools run --rm runtime \
    ./build/examples/multimodal/visual_build \
    --onnxDir /workspace/onnx/dflash-tree-base/visual \
    --engineDir "/workspace/engines/${engine_subdir}" \
    --minImageTokens 128 \
    --maxImageTokens 4096 \
    --maxImageTokensPerImage 512
fi

echo "DFlash DDTree artifacts: ${workspace_dir}/engines/${engine_subdir}"
