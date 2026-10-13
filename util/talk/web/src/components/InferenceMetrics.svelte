<script>
  import { onDestroy } from 'svelte';
  import { formatMetric, performanceTitle, responseTiming, formatElapsed } from '../lib/performance.js';
  export let performance = null;
  export let startedAt = null;
  export let running = false;
  let now = globalThis.performance.now();
  let timer;
  $: timing = responseTiming(performance);
  $: waiting = running && typeof startedAt === 'number' && !timing;
  $: {
    clearInterval(timer);
    if (waiting) {
      now = globalThis.performance.now();
      timer = setInterval(() => now = globalThis.performance.now(), 1000);
    }
  }
  onDestroy(() => clearInterval(timer));
</script>

{#if performance || waiting}
  <div class="inference-metrics" role="group" aria-label="모델 속도" class:live={waiting || performance?.live}>
    {#each ['pp', 'tg', 'ttft'] as kind}
      <span class="metric" title={performanceTitle(performance, kind)}>
        <span class="metric-label">{kind}</span>
        <span>{formatMetric(performance?.[kind], performance?.[`${kind}_estimated`], kind === 'ttft' ? 's' : 'tok/s')}</span>
      </span>
    {/each}
    {#if timing}
      <div class="response-timing-row" title="전송부터 응답 종료까지. 도구 실행·대기 시간 포함.">
        <span>소요 {timing.elapsed}</span><span>{timing.label} <time datetime={timing.datetime} title={timing.title}>{timing.clock}</time></span>
      </div>
    {:else if waiting}
      <div class="response-timing-row"><span>소요 {formatElapsed(Math.max(0, (now-startedAt)/1000))}</span><span>응답 진행 중</span></div>
    {/if}
  </div>
{/if}

<style>
  .inference-metrics { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 9px; color: var(--muted, #777); font-size: 11px; line-height: 1.5; font-variant-numeric: tabular-nums; }
  .response-timing-row { display: flex; gap: 12px; flex-wrap: wrap; width: 100%; }
  .metric { display: inline-flex; gap: 5px; white-space: nowrap; }
  .metric-label { font-weight: 600; }
  .live { opacity: .85; }
</style>
