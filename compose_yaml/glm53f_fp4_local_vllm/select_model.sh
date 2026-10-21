#!/usr/bin/env bash
: "${HF_CACHE:=${HOME}/.cache/huggingface}"
case "${MODEL_VARIANT:-abliterated}" in
 official) MODEL_REPO=nvidia/GLM-5.3-Flash-FP4-Local; MODEL_REVISION=76348ddb2be2e35e39ecfc00e75dc07703f4f3d5 ;;
 abliterated) MODEL_REPO=edp1096/Huihui-GLM-5.3-Flash-abliterated-FP4-Local; MODEL_REVISION=bbaaa926e0fc3b16fa604fa284010030f8279bb7 ;;
 *) echo 'MODEL_VARIANT must be official or abliterated' >&2; return 2 ;;
esac
MODEL_HOST_PATH="$HF_CACHE/$MODEL_REPO"
export HF_CACHE MODEL_REPO MODEL_REVISION MODEL_HOST_PATH
