# Gemma 4 E2B: LiteRT-LM vs llama.cpp on DGX Spark

Measured on 2026-08-24 through each engine's OpenAI-compatible
`/v1/chat/completions` API with the production Media prompt systems.

## Compared engines

- LiteRT-LM: current Huihui Gemma 4 E2B v1 INT8 LiteRT bundle, GPU backend.
- llama.cpp same-source control: Huihui Gemma 4 E2B v1 i1 `Q4_K_M`.
- llama.cpp candidate: Gemma 4 E2B QAT-Q4 `Q4_K`, without and with the
  model's 163 MiB MTP draft head.
- llama.cpp build: v0.2.0, CUDA 13.1, native `sm_121a`, 4,096-token context,
  one slot, all layers on GPU, flash attention enabled.

All requests used greedy decoding (`temperature=0`, `top_k=1`, seed 42,
reasoning disabled). The 33-case suite covers short and dense T2I prompts,
exact Korean/English visible text, counts and attribute binding, negations,
Control non-invention, Identity Edit preservation, inpaint/outpaint continuity,
T2V chronology/dialogue/sound, and Korean/English/Japanese subtitle translation.

## Results

| Engine | Constraint checks | First full pass mean | Warm repeat mean |
|---|---:|---:|---:|
| LiteRT v1 INT8 GPU | 322/406 (79.3%) | 2,068 ms | 2,068 ms |
| llama v1 i1 Q4_K_M | 326/406 (80.3%) | 2,141 ms | not repeated |
| llama QAT-Q4, no MTP | 335/406 (82.5%) | 2,172 ms | not repeated |
| llama QAT-Q4 + MTP | 335/406 (82.5%) | 1,316 ms | **606 ms** |

The lexical checks are deliberately strict and undercount correct paraphrases,
so they are a regression detector rather than an absolute quality score.
The same-source v1 comparison did not show a task-level Q4 quality collapse.
The QAT-Q4 candidate improved the checks most noticeably on dense prompt
preservation, multi-subject binding, Identity Edit reference roles, exact masked
text, and outpaint exclusion constraints.

A semantic A/B review of LiteRT versus QAT-Q4+MTP preferred QAT in 13 cases,
LiteRT in 6, and treated 14 as effectively tied. QAT more consistently retained
"no umbrella", "no extra people/tools", and "introduce no new objects". It also
kept Korean/English text punctuation outside the quoted text where LiteRT twice
inserted a comma inside it. LiteRT retained a few isolated negative constraints
better and was slightly preferable in one four-robot video prompt. Subtitle
translation was mostly tied; QAT avoided one unsupported `he` introduced by
LiteRT in a gender-neutral Korean sentence.

MTP produced exactly the same 33 outputs on two separate runs. Compared with
QAT-Q4 without MTP, 21/33 outputs were byte-identical and the other 12 were
semantically equivalent variants. Draft acceptance was roughly 41-60% in the
sampled tail, with logged effective generation around 160-211 tokens/s.

## Memory and disk

| Engine | Host process RSS/HWM after suite | CUDA allocation shown by nvidia-smi |
|---|---:|---:|
| LiteRT v1 INT8 | 0.56 GiB RSS / 1.45 GiB HWM | 3.72 GiB |
| llama v1 i1 Q4_K_M | 3.08 GiB | 1.87 GiB |
| llama QAT-Q4 | 3.10 GiB | 1.86 GiB |
| llama QAT-Q4 + MTP | 3.40 GiB | 2.04 GiB |

On GB10 unified memory, process RSS and CUDA allocation can refer to overlapping
physical pages and must not be added as if they were separate VRAM pools.
The candidate files occupy about 3.2 GiB plus 163 MiB for MTP. The reusable
CUDA llama.cpp image is about 1.53 GiB.

## Conclusion

For text-only Media prompt enhancement and subtitle translation, the tested
QAT-Q4+MTP llama.cpp configuration is the preferred candidate. It was not lower
quality than the current LiteRT engine in this workload, retained more hard
constraints, used modest unified memory, and reached about 0.61 seconds per
request once llama.cpp graph specialization was warm.

## Multimodal follow-up

The vision path was tested separately after the text benchmark. Eight known
images were evaluated twice: direct visual description and the production I2V
prompt system. The suite includes complex object stacks, composition and pose,
illustration-versus-photograph recognition, makeup, and OCR.

| Engine | Visual detail checks | Mean latency |
|---|---:|---:|
| LiteRT v1 INT8, JPEG | **80/110 (72.7%)** | 3,567 ms |
| llama QAT-Q4 + BF16 mmproj + MTP, JPEG | 61/110 (55.5%) | **1,047 ms** |
| llama QAT-Q4 + BF16 mmproj, no MTP, JPEG | 58/110 (52.7%) | 1,352 ms |
| llama v1 i1-Q4_K_M + matching Q8 mmproj, JPEG | 61/110 (55.5%) | 1,398 ms |

The gap was semantic, not just lexical. llama.cpp repeatedly described the rice
cat bento as pasta with a cartoon overlay, called a flat-color ocean
illustration a photograph of a male surfer on a surfboard, missed the white dog
under the large tree, misread blonde hair as dark hair, lost the side-profile
pose, and failed to read `CAT` and `KREA 2 Turbo`. LiteRT recognized most of
these details correctly. Disabling MTP and using the matching v1 projector did
not close the gap.

Both engines rejected the original WebP assets. LiteRT returned HTTP 500 and
llama.cpp returned HTTP 400; both worked after lossless-content-equivalent JPEG
conversion. Media should normalize uploaded I2V references to JPEG or PNG before
calling either backend.

The production decision is to use QAT-Q4+MTP llama.cpp for text prompt
enhancement, subtitle translation, and SparkTalk. Media does not require a
separate I2V caption pass: LTX receives its first-frame image directly, while
Krea 2 uses its own Qwen3-VL conditioning path. LiteRT can therefore be retired
despite its better result in this isolated vision benchmark.

Raw reports in this directory preserve every input, output, check, latency, API
usage block, and llama.cpp timing block.
