<script>
  import { formatMetric, performanceTitle } from '../lib/performance.js';
  export let performance = null;
</script>

{#if performance}
  <div class="inference-metrics" role="group" aria-label="모델 속도" class:live={performance.live}>
    {#each ['pp', 'tg', 'ttft'] as kind}
      <span class="metric" title={performanceTitle(performance, kind)}>
        <span class="metric-label">{kind}</span>
        <span>{formatMetric(performance[kind], performance[`${kind}_estimated`], kind === 'ttft' ? 's' : 'tok/s')}</span>
      </span>
    {/each}
  </div>
{/if}

<style>
  .inference-metrics { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 9px; color: var(--muted, #777); font-size: 11px; line-height: 1.5; font-variant-numeric: tabular-nums; }
  .metric { display: inline-flex; gap: 5px; white-space: nowrap; }
  .metric-label { font-weight: 600; }
  .live { opacity: .85; }
</style>
