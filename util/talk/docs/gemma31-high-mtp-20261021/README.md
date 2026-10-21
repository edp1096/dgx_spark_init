# Gemma 4 31B: higher MTP depths

One DGX Spark, EXL3 4-bit assistant body / 6-bit head, same target and Split-KV policy as the [main comparison](../gemma31-spec-opt-20261021). Selected: fixed MTP9; MTP15 extension cancelled. Short prompts only; median TG of three matched 512-token samples per language.

| MTP | Korean TG | Code TG | Korean acceptance | Code acceptance |
| --- | ---: | ---: | ---: | ---: |
| 9 | 32.22 | 41.12 | 32.0% | 43.5% |
| 18 | 25.06 | 32.21 | 16.7% | 22.4% |
| 21 | 24.32 | 30.95 | 14.5% | 19.2% |
| 24 | 23.35 | 29.74 | 12.7% | 16.8% |

All 8 arithmetic/JSON smoke checks passed; no replacement characters. These checks do not establish general quality. New verify widths 19/22/25 passed FP32-reference, pinned-baseline and CUDA-graph checks before serving.

Source: [measurements](summary.json), [raw requests](raw), [settings and base experiment](provenance.json), [kernel qualification](kernel-checks/high-width-results.json).
