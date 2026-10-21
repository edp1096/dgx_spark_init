# DeepSeek 1017 on TensorFold

Talk manages both ranks of the verified CUDA Zig engine over the CX7 RoCE link.
Select **DeepSeek V4.1 Flash 1017 EXL3 2bit** in Talk, then start the set.
Stop the set before changing the weight variant. Both ranks stop together.

**Abliterated (L10-35)** is the default weight variant. **Original (1017)** remains
selectable in the same weight selector; it keeps its own prepared files and state.

`MTP_TOKENS=auto` is the default: it uses the cost-based policy with depth capped at 5. Values `1` to
`5` select that fixed MTP depth. Stop the set before changing the runtime option;
both ranks use the same policy. Other serving geometry and AOT settings stay fixed.

The 2026-10-19 Korean comparison (64 requests, one active stream, thinking off)
measured 57.49 tok/s for auto and 55.51 tok/s for the best fixed depth, 2.
All four prompts produced identical output across settings and repetitions.
This result covers 256/768-token outputs with short prompts, not full 1M-context
or concurrent workloads. Raw metrics and methodology are recorded in
`util/talk/docs/ds41-mtp-korean-comparison-20261019.json`.

The profile keeps 1M context, four slots, a 2.1M shared FP8 KV pool, MTP,
native image input and tool calling. ASR and TTS run on demand on the head.
Their requests require enough memory to preserve the LLM and the memory reserve.

## Prepared layout

Each host uses its configured model cache, not a workspace or download-job path:

```text
<model_cache>/tensorfold/deepseek-v41-flash-1017/
  models/official/                 # SHA256-verified EXL3 2.8bpw pack
  models/abliterated/              # separate, verified L10-35 overlay
  engram/rankN/                   # 1017 Engram only
  assets/rankN/                   # rope, host tables, complete AOT set
  prepared/<variant>/rankN/       # never shared between weight variants
  state/<variant>/rankN/          # calibration and disk sessions
  cache/rankN/pfdense-table.json
  verification/pack.verified.json
```

The image `local/sparktalk-ds41f-tensorfold:9131e10a` must have the digest in
`env.sample` on both hosts. `manage.sh validate` checks both installations before
starting either rank. A failed partial launch stops both containers and retains logs.
The registration does not automatically download or rebuild missing artifacts.

## Rebuild after formatting

Follow the pinned upstream [installation guide](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/853310dab967480013eda7cef4e79c2fd041629a/docs/ZIG-SERVE.md).
Use recipe revision `853310dab967480013eda7cef4e79c2fd041629a` and its included
CUDA Zig engine `9131e10a`; generic TensorFold 1.1 is not a replacement.
Build the image, download the pinned pack, pack the **1017** Engram shards,
generate per-rank tables and compile all AOT variants for `profile.env`.
Verify `run-zig.sh N --check-aot` on each node before creating its verification marker.
Tag the qualified image and update the expected digest if rebuilding it.
Never reuse September Engram files or caches from different weights.

## Source

- [Engine, launcher, profile and tuning table](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/tree/853310dab967480013eda7cef4e79c2fd041629a)
- [EXL3 2.8bpw weights](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw/tree/859396591c340195526aaf61f94b4ad5323a5e18)
- [Original 1017 Engram source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)
- [Abliterated L10-35 sidecar](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated):
  `exl3_wo_b_l10_35_2.8bpw.safetensors`, SHA256
  `96cd6320e04e3074d363bfb4c3ff7e17f2bc06163f153f49ed9b3a074834f2b9`.

Upstream Apache-2.0 license and notice accompany the adapted launcher and profile.
