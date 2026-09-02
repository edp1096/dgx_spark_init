<script>
  export let state = null;
  export let open = false;
  export let loading = false;
  export let disabled = false;
  export let onToggle = () => {};
  export let onCompact = () => {};
  export let onReset = () => {};
  export let onJump = () => {};
  export let onRecall = () => {};

  $: percent = state?.input_budget > 0 ? Math.min(100, Math.round((state.estimated_tokens || 0) * 100 / state.input_budget)) : 0;
  const recallLabel = (item) => item.kind === 'user' ? '사용자 설정' : item.kind === 'memory' ? '장기 기억' : '과거 대화';
</script>

<nav class="context-rail" aria-label="컨텍스트 지도">
  <button class="context-rail-toggle" class:warning={percent >= 80} onclick={onToggle} title="컨텍스트 지도">
    <span>{percent}%</span>
  </button>
  <div class="context-marks" aria-hidden="true">
    {#each state?.segments || [] as segment}
      <button class="context-mark summarized" onclick={() => onJump(segment.start_message_id)} title={`요약 구간 ${segment.start_message_id}–${segment.end_message_id}`}></button>
    {/each}
    {#if state?.active_start_message_id}
      <button class="context-mark active" onclick={() => onJump(state.active_start_message_id)} title="현재 원문 컨텍스트"></button>
    {/if}
  </div>
</nav>

{#if open}
  <button class="context-backdrop" aria-label="컨텍스트 지도 닫기" onclick={onToggle}></button>
  <aside class="context-panel">
    <div class="context-panel-title"><div><strong>컨텍스트 지도</strong><small>화면 원본과 모델 컨텍스트를 분리해 관리합니다.</small></div><button onclick={onToggle}>×</button></div>
    {#if state}
      <div class="context-meter"><span style:width={`${percent}%`}></span></div>
      <div class="context-stats">
        <span>활성 컨텍스트 <strong>{state.estimated_tokens?.toLocaleString() || 0}</strong></span>
        <span>입력 예산 <strong>{state.input_budget?.toLocaleString() || '자동 감지 안 됨'}</strong></span>
        <span>원문 <strong>{state.active_tokens?.toLocaleString() || 0}</strong></span>
        <span>요약 <strong>{state.summary_tokens?.toLocaleString() || 0}</strong></span>
        <span>회수 <strong>{state.recall_tokens?.toLocaleString() || 0}</strong></span>
      </div>
      {#if state.notice}<p class="context-notice">{state.notice}</p>{/if}
      <div class="context-legend"><span><i class="summarized"></i>구조화 요약</span><span><i class="active"></i>현재 원문</span></div>
      {#if state.recalls?.length}
        <section class="context-recalls">
          <strong>이번 응답에 회수된 기억</strong>
          {#each state.recalls as item}
            <article>
              <span>{recallLabel(item)}{item.title ? ` · ${item.title}` : ''}</span>
              <p>{item.content}</p>
              {#if item.session_id && item.message_id}<button onclick={() => onRecall(item)}>원본 대화 열기</button>{/if}
            </article>
          {/each}
        </section>
      {/if}
      <div class="context-segments">
        {#each state.segments || [] as segment, index}
          <details>
            <summary><span>구간 {index + 1}</span><small>메시지 {segment.start_message_id}–{segment.end_message_id} · 약 {segment.estimated_tokens?.toLocaleString()}토큰</small></summary>
            <pre>{segment.summary}</pre>
            <button onclick={() => onJump(segment.start_message_id)}>원본 위치로 이동</button>
          </details>
        {/each}
        <div class="context-active-card">
          <strong>현재 원문 구간</strong>
          <small>메시지 {state.active_start_message_id || '–'}–{state.active_end_message_id || '–'}</small>
          {#if state.active_start_message_id}<button onclick={() => onJump(state.active_start_message_id)}>원본 위치로 이동</button>{/if}
        </div>
      </div>
      <div class="context-actions">
        <button onclick={onReset} disabled={disabled || loading || !state.segments?.length}>요약 초기화</button>
        <button class="primary" onclick={onCompact} disabled={disabled || loading}>{loading ? '정리 중…' : '지금 구간 정리'}</button>
      </div>
    {:else}<p class="context-notice">컨텍스트 정보를 불러오는 중입니다.</p>{/if}
  </aside>
{/if}
