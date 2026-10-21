#!/usr/bin/env bash
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
trial_dir=${VELO_YARN_WORKDIR:-"${XDG_CACHE_HOME:-$HOME/.cache}/model-download-jobs/velogb10-yarn-20261009"}
revision=a6ad23d60e082ff9cca2bb78adda4ecee771388c
if [[ ! -d "$trial_dir/source/.git" ]]; then
    mkdir -p "$trial_dir"
    git clone --filter=blob:none --no-checkout https://github.com/sf-stav/veloGB10.git "$trial_dir/source"
    git -C "$trial_dir/source" checkout --detach "$revision"
fi
if [[ -x "$trial_dir/cargo/bin/cargo" ]]; then
    export CARGO_HOME="$trial_dir/cargo"
    export RUSTUP_HOME="$trial_dir/rustup"
    export PATH="$CARGO_HOME/bin:$PATH"
fi
command -v cargo >/dev/null || { printf '%s\n' 'Rust stable toolchain is required.' >&2; exit 1; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { printf '%s\n' 'nvcc is required to rebuild the changed CUDA ABI.' >&2; exit 1; }
cd "$trial_dir/source"
[[ $(git rev-parse HEAD) == "$revision" ]] || { printf '%s\n' 'Wrong Velo revision; use an isolated checkout.' >&2; exit 1; }
if ! git apply --reverse --check "$recipe_dir/velogb10-exl3-schema.patch" 2>/dev/null; then
    if git apply --check "$recipe_dir/velogb10-exl3-yarn.patch" 2>/dev/null; then
        git apply "$recipe_dir/velogb10-exl3-yarn.patch"
    else
        git apply --reverse --check "$recipe_dir/velogb10-exl3-yarn.patch"
    fi
    git apply --check "$recipe_dir/velogb10-exl3-schema.patch"
    git apply "$recipe_dir/velogb10-exl3-schema.patch"
fi
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
cargo build --release --locked -j "${VELO_BUILD_JOBS:-8}"
cargo test --release --locked --lib exl3_rope -- --nocapture
for filter in json_schema::tests exl3_schema::tests tokenizer::tests server:: exl3_serve::; do
    cargo test --release --locked --lib "$filter"
done
if [[ -n ${VELO_TOKENIZER_PATH:-} ]]; then
    cargo test --release --locked --test exl3_schema_tokenizer -- --ignored --nocapture
fi
cargo test --release --locked --test exl3_schema_gpu -- --ignored --nocapture --test-threads=1
cargo test --release --locked --test exl3_yarn_gpu -- --ignored --nocapture --test-threads=1
