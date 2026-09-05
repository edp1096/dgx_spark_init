<script>
  export let runtime = null;
  export let busy = false;
  export let onAction = async () => {};
  export let onRefresh = async () => {};

  let targetBundle = '';
  let lastSelectedBundle = '';
  $: selectedBundle = runtime?.bundles?.find((bundle) => bundle.id === runtime.selected_bundle) || runtime?.bundles?.[0];
  $: if (selectedBundle?.id && selectedBundle.id !== lastSelectedBundle) {
    lastSelectedBundle = selectedBundle.id;
    targetBundle = selectedBundle.id;
  }
  $: selectedComponents = (runtime?.components || []).filter((component) => selectedBundle?.components?.includes(component.id));
  $: operationRunning = runtime?.operation?.state === 'running';
  $: selectedBundleOnline = selectedComponents.length > 0 && selectedComponents.every((component) => component.health === 'online');
  $: targetIsSelected = targetBundle === runtime?.selected_bundle;
  $: primaryLabel = operationRunning ? '기동 중' : !targetIsSelected ? '전환' : selectedBundleOnline ? '실행 중' : '복구';
  $: memoryPercent = runtime?.memory?.total_gib ? Math.min(100, Math.max(0, runtime.memory.used_gib / runtime.memory.total_gib * 100)) : 0;
  $: operationSteps = (runtime?.operation?.steps || []).slice(-7);
  $: operationETA = /^\d+(?::\d+){1,2}$/.test(runtime?.operation?.eta || '') ? runtime.operation.eta : '';

  function formatGiB(value) {
    return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)} GiB` : '—';
  }

  function stateLabel(component) {
    if (component.health === 'online') return '온라인';
    if (component.health === 'starting') return '준비 중';
    if (component.health === 'failed') return '실패';
    if (component.status === 'exited') return '중지됨';
    if (component.status === 'missing') return '미설치';
    return '오프라인';
  }

  function formatElapsed(value) {
    if (!value) return '';
    const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1000));
    if (seconds < 60) return `${seconds}초`;
    const minutes = Math.floor(seconds / 60);
    return `${minutes}분 ${seconds % 60}초`;
  }
</script>

<section class="runtime-panel" aria-label="DGX Spark 운영">
  <div class="runtime-heading">
    <div><strong>{selectedBundle?.name || 'AI 세트'}</strong><small>{selectedBundle?.description || '상태 확인 중'}</small></div>
    <button type="button" onclick={onRefresh} title="운영 상태 새로고침" aria-label="운영 상태 새로고침">↻</button>
  </div>

  {#if runtime?.memory}
    <div class="runtime-memory">
      <div><span>통합메모리</span><b>{formatGiB(runtime.memory.used_gib)} / {formatGiB(runtime.memory.total_gib)}</b></div>
      <div class="runtime-memory-track"><i style={`width:${memoryPercent}%`}></i></div>
      <small>시스템 가용 {formatGiB(runtime.memory.available_gib)} · 즉시 여유 {formatGiB(runtime.memory.free_gib)}</small>
    </div>
  {/if}

  {#if operationRunning}
    <div class="runtime-operation">
      <div><strong>{runtime.operation.phase || '처리 중'}</strong><b>{Math.round((runtime.operation.progress || 0) * 100)}%</b></div>
      <div class="runtime-progress"><i style={`width:${Math.round((runtime.operation.progress || 0) * 100)}%`}></i></div>
      {#if runtime.operation.detail}<p>{runtime.operation.detail}</p>{/if}
      <small>{operationETA ? `예상 ${operationETA} 남음` : `경과 ${formatElapsed(runtime.operation.started_at)} · 로그를 계속 확인 중`}</small>
      {#if operationSteps.length > 1}
        <ol class="runtime-steps" aria-label="기동 단계">
          {#each operationSteps as step}
            <li class:current={step.state === 'current'} class:failed={step.state === 'failed'}>
              <i>{step.state === 'complete' ? '✓' : step.state === 'failed' ? '!' : ''}</i>
              <span><b>{step.phase}</b>{#if step.detail}<small>{step.detail}</small>{/if}</span>
            </li>
          {/each}
        </ol>
      {/if}
    </div>
  {:else if runtime?.operation?.state === 'failed'}
    <p class="runtime-error">{runtime.operation.error}</p>
  {/if}

  <div class="runtime-components">
    {#each selectedComponents as component}
      <div class="runtime-component">
        <span class:online={component.health === 'online'} class:starting={component.health === 'starting'} class:failed={component.health === 'failed'}><i></i><span><b>{component.name}</b><small>{component.phase || component.model || component.role} · {component.host || 'local'}</small></span></span>
        <div><b>{stateLabel(component)}</b><small>{component.gpu_memory_gib ? formatGiB(component.gpu_memory_gib) : ''}</small></div>
      </div>
    {/each}
  </div>

  <div class="runtime-switch">
    <select bind:value={targetBundle} aria-label="전환할 AI 세트">
      {#each runtime?.bundles || [] as bundle}
        <option value={bundle.id}>{bundle.name} · 약 {formatGiB(bundle.memory_gib)}</option>
      {/each}
    </select>
    <button type="button" class="primary" disabled={busy || operationRunning || !targetBundle || (targetIsSelected && selectedBundleOnline)} onclick={() => onAction('start', targetBundle)}>{primaryLabel}</button>
    <button type="button" disabled={busy || operationRunning || !runtime?.selected_bundle} onclick={() => onAction('stop', runtime.selected_bundle)}>중지</button>
  </div>
  {#each Object.entries(runtime?.hosts || {}) as [host, status]}
    <small class="runtime-docker" class:offline={!!status.error}>{host} · {status.error ? '연결 실패: ' + status.error : '가용 ' + formatGiB(status.memory.available_gib) + ' / 즉시 여유 ' + formatGiB(status.memory.free_gib)}</small>
  {/each}
  <small class="runtime-docker" class:offline={runtime?.docker !== 'online'}>Docker · {runtime?.docker === 'online' ? '정상' : '연결 오류'}</small>
</section>
