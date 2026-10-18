# Huihui GLM-5.3 Flash FP4-Local

Unofficial derivative of NVIDIA `GLM-5.3-Flash-FP4-Local` with Huihui Q8 deltas applied.

[Download on Hugging Face](https://huggingface.co/edp1096/Huihui-GLM-5.3-Flash-abliterated-FP4-Local) · [Publication record](publication-result.json)

## Sources

| Role | Source | Pinned revision |
| --- | --- | --- |
| Base FP4-Local checkpoint | [nvidia/GLM-5.3-Flash-FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/tree/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5) | `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` |
| Abliteration donor (Q8) | [huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF/tree/50d39f500cbd1d0478295da89d417607521d378b) | `50d39f500cbd1d0478295da89d417607521d378b` |
| Delta reference (Q8) | [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9) | `621d456e93e926e4b52f85cff5f634358c1828f9` |

Applied delta: `DQ(Huihui Q8) - DQ(Unsloth Q8)`. The conversion used these exact revisions.

## Model

- Location on host 60: `~/.cache/huggingface/edp1096/Huihui-GLM-5.3-Flash-abliterated-FP4-Local`
- Size: about 165 GiB across 42 weight files
- Changed: 44 attention-output/shared-expert-down matrices in layers 15–36, plus 22 scale tensors
- Preserved: all other tensors, including routed experts, CSF scales, MTP, vision, and tokenizer files

## Runtime

Requires two DGX Sparks (TP2) and B12X vLLM with CSF support.
Exact tested settings are in the model directory's `runtime-configuration.json`.

```text
PYTORCH_ALLOC_CONF=backend:native,garbage_collection_threshold:0.7
```

Cold-start loading caused memory pressure and some host swapping.

## Validation

- Full weight and untouched-byte checks passed; hashes matched on both nodes
- 8/8 smoke checks passed: arithmetic, Korean, code, JSON, tools, vision, long-context retrieval, and post-long-context response
- Longest tested input: 64,481 tokens. Configured limit: 1,048,576; actual 1M requests were not tested

See [validation results](qualification-result.json) for evidence and memory measurements.

## Conversion tools

1. `download.py` / `fetch_deltas.py`: fetch pinned inputs and Q8 ranges; verify hashes
2. `build.py --mode analyze` → `--mode build`: transfer deltas, requantize in the existing FP8 formats, and verify
3. `qualify.py`: test a separately launched server

Pinned revisions, numerical errors, and output hashes are recorded in the model directory's `transfer-manifest.json`.
No additional QAD training or activation calibration was performed. Huihui BF16 equivalence and preservation of upstream benchmark scores are not guaranteed.
