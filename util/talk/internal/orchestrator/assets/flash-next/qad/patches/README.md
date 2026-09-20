# QSA FP8 patch provenance

Source: sgl-project/sglang PR 36644, head `3df8e1e7dbc5807696622afe2929b6c33c185ca3`, base `4ccff141dbe992794f9da6c3aa23535b4f72000d`.
Repository license: Apache-2.0 (SGLang). Runtime-only hunks retained;
upstream GPU tests are in `experiments/lil-qad-tp1/test_qsa_fp8_upstream.py`.
Runtime patch SHA256: `c4e00cde8ca80818fbf8cd9c91e9becf550127dace48f659185d4c63f88bdedb`.

This patch preserves FP8 KV storage while dequantizing selected K/V to the
query dtype for attention, including non-unit descales. It does not claim
native FP8 tensor-core attention. The local SM121 KDA path is preserved.
