# Speculative Decoding Benchmark — glm-5.3-flash

- **Run ID**: `2026-09-27T16-00-28.971383Z_6caaf594`
- **Date**: `2026-09-27T16:00:28.971502+00:00`
- **Mode**: spec-bench
- **Spec Method**: dflash

## Results

| Prompt | Depth | Eff t/s | Stream t/s | α (accept) | Waste | τ (length) | Window | Draft t/s | Speedup | TTFT (ms) | Total (ms) | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filler | 0 | 36.0 | 35.8 | 55.3% | 45% | 3.8 | 5 | 47.9 | — | 1,993 | 5,546 | 128 |
| code | 0 | 43.4 | 43.0 | 73.6% | 26% | 4.7 | 5 | 47.4 | — | 478 | 3,430 | 128 |
| structured | 0 | 50.1 | 49.7 | 87.5% | 12% | 5.4 | 5 | 46.9 | — | 585 | 3,141 | 128 |
| filler | 2048 | 32.4 | 32.1 | 47.4% | 53% | 3.4 | 5 | 48.1 | — | 4,251 | 8,204 | 128 |
| code | 0 | 44.9 | 44.6 | 74.8% | 25% | 4.7 | 5 | 47.4 | — | 483 | 3,331 | 128 |
| structured | 0 | 46.9 | 46.6 | 88.3% | 12% | 5.4 | 5 | 44.0 | — | 498 | 3,226 | 128 |
| filler | 8192 | 32.5 | 32.2 | 47.4% | 53% | 3.4 | 5 | 48.2 | — | 5,266 | 9,207 | 128 |
| code | 0 | 48.6 | 48.2 | 84.8% | 15% | 5.2 | 5 | 47.4 | — | 474 | 3,110 | 128 |
| structured | 0 | 50.1 | 49.8 | 88.3% | 12% | 5.4 | 5 | 47.0 | — | 586 | 3,138 | 128 |

## Acceptance Rate by Prompt Type

```
      filler d0     ██████████████████████░░░░░░░░░░░░░░░░░░ 55.3%
        code d0     █████████████████████████████░░░░░░░░░░░ 73.6%
  structured d0     ███████████████████████████████████░░░░░ 87.5%
      filler d2048  ██████████████████░░░░░░░░░░░░░░░░░░░░░░ 47.4%
        code d0     █████████████████████████████░░░░░░░░░░░ 74.8%
  structured d0     ███████████████████████████████████░░░░░ 88.3%
      filler d8192  ██████████████████░░░░░░░░░░░░░░░░░░░░░░ 47.4%
        code d0     █████████████████████████████████░░░░░░░ 84.8%
  structured d0     ███████████████████████████████████░░░░░ 88.3%
```

## Per-Prompt-Type Summary

| Prompt Type | Avg Eff t/s | Avg Stream t/s | Avg α | Avg Waste | Avg Draft t/s |
|---|---:|---:|---:|---:|---:|
| code | 45.6 | 45.3 | 77.7% | 22% | 47.4 |
| filler | 33.6 | 33.4 | 50.0% | 50% | 48.0 |
| structured | 49.1 | 48.7 | 88.1% | 12% | 46.0 |

## Draft Efficiency

| Metric | Value |
|---|---|
| Avg Draft Window | 5 tokens/step |
| Avg Acceptance Length (τ) | 4.6 tokens/step |
| Window Utilization | 92% |
| Avg Waste | 28% |

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
