#!/bin/sh
set -eu
revision=e8800c8def63449808a4092798442652ed460552
mkdir -p /src
cd /src
git init -q
git remote add origin https://github.com/edwardkim/rhwp.git
git config remote.origin.promisor true
git config remote.origin.partialclonefilter blob:none
git sparse-checkout init --no-cone
git sparse-checkout set --no-cone '/Cargo.toml' '/Cargo.lock' '/build.rs' '/rust-toolchain.toml' '/LICENSE' '/saved/blank2010.hwp' '/examples/pr599_png_gateway.rs' '/tests/' '/src/' '/crates/' '/assets/' '/bindings/Native/' '/tools/rhwp-subsecond/' '/tools/batch-convert/' '/tools/llm_verifier/verdict_protocol/Cargo.toml' '/tools/llm_verifier/verdict_protocol/src/' '/tools/llm_verifier/claim_bind/Cargo.toml' '/tools/llm_verifier/claim_bind/src/' '/tools/llm_verifier/criteria_decomp/Cargo.toml' '/tools/llm_verifier/criteria_decomp/src/'
git fetch -q --depth 1 --filter=blob:none origin "$revision"
git checkout -q --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "$revision"
