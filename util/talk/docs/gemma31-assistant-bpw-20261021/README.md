# Gemma 4 31B assistant bitrate comparison

Measured on one DGX Spark on 2026-10-21. The target is the same Huihui Gemma 4 31B NVFP4 checkpoint in every arm. The assistant performs speculative decoding; these assistant checkpoints are not standalone chat models.

| Assistant | Weight MiB | Korean TG tok/s | Code TG tok/s | Korean draft acceptance | Code draft acceptance | 9,041-token PP tok/s | Basic checks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Google original BF16 | 895.5 | 25.99 | 32.26 | 39.0% | 53.2% | 858.4 | 10/10 |
| EXL3 body4 / head6 | 288.8 | 30.34 | 37.37 | 38.7% | 50.0% | 861.9 | 10/10 |
| EXL3 body3 / head6 | 264.9 | 29.45 | 37.13 | 37.2% | 50.8% | 856.7 | 10/10 |
| EXL3 body2 / head6 | 240.9 | 28.91 | 32.71 | 35.6% | 43.3% | 861.6 | 10/10 |
| No assistant | 0.0 | 10.40 | 10.39 | n/a | n/a | 766.7 | 10/10 |

TG is the median of three Korean and three coding prompts per arm, each generating exactly 512 tokens. These are three matched prompt variants, not repeated measurements of one prompt. Temperature is zero and the sampling seed is 42. Warmup and model loading are excluded. Prefix caching is disabled, and every recorded request reports zero cached tokens. No speed sample contains an end-of-turn/EOS marker.

PP uses the engine's `prefill_finished_time - forward_entry_time`; TG uses streamed completion tokens after the first nonempty chunk divided by time after that chunk. The first chunk's actual token count is subtracted. The raw engine timing fields and all streaming event times are retained. Draft acceptance is the total correct proposed drafts divided by total proposed drafts (seven drafts per verification), excluding the bonus token. PP at 9K is a single measurement per arm and is not a statistically ranked PP benchmark.

## Interpretation

EXL3 body3 / head6 changes median TG by -3.0% for Korean and -0.6% for code relative to body4/head6, saving 24.0 MiB of assistant weights.
EXL3 body2 / head6 changes median TG by -4.7% for Korean and -12.5% for code relative to body4/head6, saving 47.9 MiB of assistant weights.

Body4/head6 changes median TG by +16.8% for Korean and +15.8% for code relative to Google original BF16.
Body4/head6 changes median TG by +191.7% for Korean and +259.6% for code relative to No assistant.

Small differences can reflect run order, thermal state, and differences in generated text. This is a bounded local comparison, not evidence that a bitrate universally wins.

## Correctness and scope

The ten basic checks cover arithmetic, JSON content, Korean instructions, Japanese translation, probability, tool-call syntax, two Python functions executed in isolated containers, and recall from 2,591- and 9,041-token prompts. The JSON content check accepts Markdown fences; it is not a strict raw-JSON conformance test. These checks do not establish general model accuracy or lossless speculative decoding. Greedy generated text can differ even for the original assistant relative to no speculation.

| Assistant | Exact quality-case text vs original assistant | Exact quality-case text vs no assistant | Replacement characters |
|---|---:|---:|---:|
| Google original BF16 | 10/10 | 9/10 | 0 |
| EXL3 body4 / head6 | 10/10 | 9/10 | 0 |
| EXL3 body3 / head6 | 9/10 | 9/10 | 0 |
| EXL3 body2 / head6 | 9/10 | 9/10 | 0 |
| No assistant | 9/10 | 10/10 | 0 |

The KV/context capacity is 262,144 tokens, but the largest evaluated prompt in this experiment is 9,041 tokens. This is not a new 256K performance or recall validation. Results use the same SGLang commit, target checkpoint, TP1, BF16 KV, chunked prefill 512, static memory fraction 0.55, SWA ratio 0.006, and one active request. MTP arms use Frozen-KV MTP with seven steps/eight draft tokens. The no-assistant arm omits speculation. Seven steps are a controlled comparison setting inherited from an earlier VeloGB10/EXL3-target sweep over 3, 7 and 15; this experiment does not establish the optimal step count for SGLang/NVFP4 or each assistant bitrate.

## Conversion and artifacts

The 22 assistant body matrices were quantized from the pinned Google BF16 source using ExLlamaV3 1.5.4's activation-free matrix quantizer, with the existing conversion procedure and seeds. No calibration or fine-tuning was performed. Both new variants copy the existing 6-bit packed output head byte-for-byte. Original-precision norm/scalar shards are also byte-identical across the three EXL3 variants. The labels 2bit/3bit describe the body, not the full assistant's average bitrate.

Mean per-matrix body NMSE is 0.068512 (2bit), 0.017431 (3bit), and 0.004468 (4bit). Matrix reconstruction error is not an accuracy score. Every shard header, index entry, packed bit width, matrix SHA256, head SHA256, and norm SHA256 was checked. Source downloads were pinned and checked against Hugging Face LFS hashes.

New weights and full server logs remain in `/home/edp1096/.cache/model-download-jobs/gemma31-assistant-bpw-20261021-174321`. Models are in `models/b2` and `models/b3`. No Talk set was enabled or changed and no new weights were uploaded.

Raw results, prompt token IDs, conversion verification, source revisions, and quality checks are attached in this directory.

## Sources

- [Google original assistant](https://huggingface.co/google/gemma-4-31B-it-assistant/tree/627c5ec1458b9086b841a91e0512fd31fd2fbbf1)
- [Existing EXL3 body4/head6 weights](https://huggingface.co/edp1096/gemma-4-31B-it-assistant-exl3/tree/359650d0d9386fbe42b820c74eefa8075017ebc6)
- [Fixed NVFP4 target](https://huggingface.co/lyf/Huihui-gemma-4-31B-it-abliterated-v2-NVFP4/tree/985633e8001d0bfd9b621738234394744b992907)
- [Pinned SGLang assistant implementation](https://github.com/sgl-project/sglang/blob/2ef0fe46693d1b88271943de5eb49a2848a512d8/python/sglang/srt/models/gemma4_mtp.py)
- [Pinned ExLlamaV3 matrix quantizer](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/modules/quant/exl3_lib/quantize.py)
- [Archived SGLang adapter and recipe](https://github.com/edp1096/dgx_spark_init/tree/main/compose_yaml/zzz_not_use/gemma4_31b_sglang)
