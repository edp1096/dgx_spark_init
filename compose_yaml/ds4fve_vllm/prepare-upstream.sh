#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
upstream="$script_dir/upstream"
if [[ ! -f "$upstream/start-deepseek-v4-flash-dspark.sh" ]]; then
 [[ ! -e "$upstream" ]] || { echo 'Incomplete upstream directory; move it aside before retrying.' >&2; exit 1; }
 git init -q "$upstream"
 git -C "$upstream" remote add origin https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
 git -C "$upstream" fetch -q --depth 1 origin b1b8dfb84d855a166a05f32a90c269118b208987
 git -C "$upstream" checkout -q --detach FETCH_HEAD
fi
patch_file="$script_dir/backports/20260906/runtime.patch"
if git -C "$upstream" apply --check "$patch_file" 2>/dev/null; then
 git -C "$upstream" apply "$patch_file"
elif ! git -C "$upstream" apply --reverse --check "$patch_file" 2>/dev/null; then
 echo 'Upstream does not match the pinned image-limit/C128A backport.' >&2; exit 1
fi
for file in patches/hotfix-vllm-c128a-prefill-cache.py scripts/test-c128a-prefill-cache.py scripts/fixtures/c128a-prefill-cache/flashinfer-prefill.py; do
 mkdir -p "$(dirname "$upstream/$file")"
 cp "$script_dir/backports/20260906/$file" "$upstream/$file"
done
