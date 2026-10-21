# Speculative Decoding Benchmark — glm-5.3-flash

- **Run ID**: `2026-09-27T14-59-13.899512Z_5aacaa44`
- **Date**: `2026-09-27T14:59:13.899604+00:00`
- **Mode**: spec-bench
- **Spec Method**: dflash

## Results

| Prompt | Depth | Eff t/s | Stream t/s | α (accept) | Waste | τ (length) | Window | Draft t/s | Speedup | TTFT (ms) | Total (ms) | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filler | 0 | 32.1 | 31.9 | 48.9% | 51% | 3.4 | 5 | 47.7 | — | 1,989 | 5,974 | 128 |
| code | 0 | 44.2 | 43.8 | 77.8% | 22% | 4.9 | 5 | 46.6 | — | 526 | 3,423 | 128 |
| structured | 0 | 50.1 | 49.7 | 88.3% | 12% | 5.4 | 5 | 46.9 | — | 499 | 3,055 | 128 |
| filler | 2048 | 27.5 | 27.3 | 36.4% | 64% | 2.8 | 5 | 48.4 | — | 4,255 | 8,908 | 128 |
| code | 0 | 40.2 | 39.9 | 66.7% | 33% | 4.3 | 5 | 47.1 | — | 473 | 3,656 | 128 |
| structured | 0 | 52.4 | 52.0 | 93.0% | 7% | 5.7 | 5 | 47.1 | — | 579 | 3,021 | 128 |
| filler | 8192 | 32.4 | 32.1 | 47.9% | 52% | 3.4 | 5 | 48.1 | — | 5,304 | 9,257 | 128 |
| code | 0 | 36.8 | 36.5 | 60.0% | 40% | 4.0 | 5 | 47.4 | — | 472 | 3,953 | 128 |
| structured | 0 | 50.3 | 50.0 | 88.3% | 12% | 5.4 | 5 | 47.2 | — | 499 | 3,042 | 128 |

## Acceptance Rate by Prompt Type

```
      filler d0     ███████████████████░░░░░░░░░░░░░░░░░░░░░ 48.9%
        code d0     ███████████████████████████████░░░░░░░░░ 77.8%
  structured d0     ███████████████████████████████████░░░░░ 88.3%
      filler d2048  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 36.4%
        code d0     ██████████████████████████░░░░░░░░░░░░░░ 66.7%
  structured d0     █████████████████████████████████████░░░ 93.0%
      filler d8192  ███████████████████░░░░░░░░░░░░░░░░░░░░░ 47.9%
        code d0     ████████████████████████░░░░░░░░░░░░░░░░ 60.0%
  structured d0     ███████████████████████████████████░░░░░ 88.3%
```

## Per-Prompt-Type Summary

| Prompt Type | Avg Eff t/s | Avg Stream t/s | Avg α | Avg Waste | Avg Draft t/s |
|---|---:|---:|---:|---:|---:|
| code | 40.4 | 40.1 | 68.1% | 32% | 47.0 |
| filler | 30.7 | 30.4 | 44.4% | 56% | 48.0 |
| structured | 50.9 | 50.6 | 89.9% | 10% | 47.1 |

## Draft Efficiency

| Metric | Value |
|---|---|
| Avg Draft Window | 5 tokens/step |
| Avg Acceptance Length (τ) | 4.4 tokens/step |
| Window Utilization | 87% |
| Avg Waste | 33% |

## Interpretation Guide

- **Eff t/s** (Effective t/s): Output tokens ÷ wall-clock generation time. This is what users experience. Higher is better.
- **Stream t/s**: Token generation rate measured from SSE stream timing. For standard decoding, this matches Eff t/s. For spec decode, Eff t/s is typically higher.
- **α (accept)**: Acceptance rate — % of draft tokens accepted by the verifier. Higher means the draft model/MTP heads predict well for this workload.
- **Waste**: Fraction of drafted tokens rejected (1 − α). Lower is better. High waste means the draft model is poorly aligned with the target.
- **τ (length)**: Average acceptance length — tokens accepted per speculative step. Higher means more tokens generated per verification pass.
- **Window**: Average tokens drafted per speculative step (the configured draft window). Compare with τ to see window utilization.
- **Draft t/s**: Rate at which draft tokens are generated, regardless of acceptance. Compare with Eff t/s to see draft overhead.
- **Speedup**: Effective t/s ÷ baseline t/s. Values > 1.0x indicate spec decode is providing a benefit.

> [!TIP]
> Acceptance rates vary significantly by prompt type. Code and structured tasks
> typically show higher acceptance rates than creative/open-ended generation
> because future tokens are more predictable.
