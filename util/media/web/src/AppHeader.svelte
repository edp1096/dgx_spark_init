<script>
  import SparkBolt from './SparkBolt.svelte'
  import { engineMeta, imageModeMeta } from './lib/catalogs.js'

  export let colorTheme
  export let engineAggregate
  export let engineAggregateLabel
  export let engineStates
  export let imageForm
  export let mobileEngineOpen
  export let monitoredEngineStatuses
  export let recognitionForm
  export let systemUsage
  export let tab
  export let toggleColorTheme
</script>

<header>
  <div><span class="mark"><SparkBolt label="Spark Media" /></span><h1>Spark Media</h1><button type="button" class="theme-toggle" title={colorTheme === 'light' ? '다크 모드로 전환' : '라이트 모드로 전환'} aria-label={colorTheme === 'light' ? '다크 모드로 전환' : '라이트 모드로 전환'} aria-pressed={colorTheme === 'light'} onclick={toggleColorTheme}>{colorTheme === 'light' ? '☾' : '☀'}</button></div>
  <div class="engine-strip">
    <span class="system-usage" title="5초 간격으로 갱신되는 DGX Spark 사용률"><b>CPU</b> {systemUsage.cpu_percent ?? '–'}% <b>GPU</b> {systemUsage.gpu_percent ?? '–'}% <b>MEM</b> {systemUsage.mem_used_gb == null ? '–' : Number(systemUsage.mem_used_gb).toFixed(1)}/{systemUsage.mem_total_gb == null ? '–' : Number(systemUsage.mem_total_gb).toFixed(1)}GB({systemUsage.mem_percent ?? '–'}%)</span>
    {#if tab === 'image'}
      <span class:running={engineStates[imageModeMeta[imageForm.mode].engine] === 'online'}><i></i>{imageModeMeta[imageForm.mode].short} API<span class="engine-state-text"> · {engineStates[imageModeMeta[imageForm.mode].engine] || 'offline'}</span></span>
      <span class:running={engineStates.prompt === 'online'}><i></i>Enhancer API<span class="engine-state-text"> · {engineStates.prompt || 'offline'}</span></span>
      <span class:running={engineStates.upscale === 'online'}><i></i>Upscale API<span class="engine-state-text"> · {engineStates.upscale || 'offline'}</span></span>
      <span class:running={engineStates.garment === 'online'}><i></i>Garment API<span class="engine-state-text"> · {engineStates.garment || 'offline'}</span></span>
      <span class:running={engineStates.faceswap === 'online'}><i></i>ReActor API<span class="engine-state-text"> · {engineStates.faceswap || 'offline'}</span></span>
    {:else if engineMeta[tab]}
      <span class:running={engineStates[engineMeta[tab][0]] === 'online'}><i></i>{engineMeta[tab][1]} API<span class="engine-state-text"> · {engineStates[engineMeta[tab][0]] || 'offline'}</span></span>
    {/if}
    {#if tab === 'video'}
      <span class:running={engineStates.prompt === 'online'}><i></i>Enhancer API<span class="engine-state-text"> · {engineStates.prompt || 'offline'}</span></span>
    {/if}
    {#if tab === 'recognition'}
      <span class:running={engineStates.recognition === 'online'}><i></i>ASR API<span class="engine-state-text"> · {engineStates.recognition || 'offline'}</span></span>
      {#if recognitionForm.translation_mode !== 'none'}<span class:running={engineStates.prompt === 'online'}><i></i>Translator API<span class="engine-state-text"> · {engineStates.prompt || 'offline'}</span></span>{/if}
    {/if}
  </div>
  <div class="mobile-engine-area">
    <span class="mobile-system-usage" title={`MEM ${systemUsage.mem_used_gb == null ? '–' : Number(systemUsage.mem_used_gb).toFixed(1)}/${systemUsage.mem_total_gb == null ? '–' : Number(systemUsage.mem_total_gb).toFixed(1)}GB(${systemUsage.mem_percent ?? '–'}%)`}>C {systemUsage.cpu_percent ?? '–'}% · G {systemUsage.gpu_percent ?? '–'}% · M {systemUsage.mem_used_gb == null ? '–' : Number(systemUsage.mem_used_gb).toFixed(1)}/{systemUsage.mem_total_gb == null ? '–' : Number(systemUsage.mem_total_gb).toFixed(1)}GB({systemUsage.mem_percent ?? '–'}%)</span>
    <button type="button" class="mobile-engine-summary {engineAggregate}" aria-expanded={mobileEngineOpen} aria-label={`API 상태: ${engineAggregateLabel}`} onclick={() => mobileEngineOpen = !mobileEngineOpen}><i></i><span>API</span></button>
    {#if mobileEngineOpen}
      <button type="button" class="mobile-engine-dismiss" aria-label="API 상태 닫기" onclick={() => mobileEngineOpen = false}></button>
      <section class="mobile-engine-popover" aria-label="각 API 상태">
        <header><strong>API 상태</strong><span class={engineAggregate}>{engineAggregateLabel}</span></header>
        <div>{#each monitoredEngineStatuses as item}<p class:online={item.online}><i></i><span>{item.label}</span><small>{item.online ? '정상' : '오프라인'}</small></p>{/each}</div>
      </section>
    {/if}
  </div>
</header>
