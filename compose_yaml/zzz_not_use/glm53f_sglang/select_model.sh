#!/usr/bin/env bash
# Source after .env. Use separate, stable directories on both hosts.
: "${HF_CACHE:=${HOME}/.cache/huggingface}"
case "${MODEL_VARIANT:-official}" in
 official) MODEL_HOST_PATH="$HF_CACHE/nvidia/GLM-5.3-Flash-NVFP4" ;;
 abliterated) MODEL_HOST_PATH="$HF_CACHE/edp1096/Huihui-GLM-5.3-Flash-abliterated-NVFP4" ;;
 *) echo 'MODEL_VARIANT must be official or abliterated' >&2; return 2 ;;
esac
DRAFT_MODEL_HOST_PATH="$HF_CACHE/incoai/GLM-5.3-Flash-DFlash2"
export HF_CACHE MODEL_HOST_PATH DRAFT_MODEL_HOST_PATH
