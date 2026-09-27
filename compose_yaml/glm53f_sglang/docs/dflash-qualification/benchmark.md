# Speculative Decoding Benchmark — glm-5.3-flash

- **Run ID**: `2026-09-27T16-38-18.855409Z_ae909826`
- **Date**: `2026-09-27T16:38:18.855539+00:00`
- **Mode**: spec-bench
- **Spec Method**: dflash

> [!NOTE]
> Acceptance rate metrics were not available from the server.
> Effective t/s (wall-clock based) still captures the real benefit
> of MTP / speculative decoding.

## Results

| Prompt | Depth | Eff t/s | Stream t/s | Speedup | TTFT (ms) | Total (ms) | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| filler | 0 | 30.1 | 29.8 | — | 1,299 | 5,557 | 128 |
| code | 0 | 43.0 | 42.7 | — | 386 | 3,363 | 128 |
| structured | 0 | 49.7 | 49.3 | — | 334 | 2,908 | 128 |
| filler | 2048 | 31.1 | 30.9 | — | 2,448 | 6,558 | 128 |
| code | 0 | 42.8 | 42.5 | — | 475 | 3,466 | 128 |
| structured | 0 | 49.7 | 49.3 | — | 350 | 2,926 | 128 |
| filler | 8192 | 29.5 | 29.2 | — | 4,805 | 9,148 | 128 |
| code | 0 | 42.8 | 42.5 | — | 476 | 3,467 | 128 |
| structured | 0 | 51.4 | 51.0 | — | 591 | 3,083 | 128 |

## Per-Prompt-Type Summary

| Prompt Type | Avg Eff t/s | Avg Stream t/s | Avg α | Avg Waste |
|---|---:|---:|---:|---:|
| code | 42.9 | 42.5 | — | — |
| filler | 30.2 | 30.0 | — | — |
| structured | 50.3 | 49.9 | — | — |

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
