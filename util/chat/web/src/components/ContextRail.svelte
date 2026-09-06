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

  $: percent = state?.input_budget > 0 ? Math.round((state.estimated_tokens || 0) * 100 / state.input_budget) : 0;
  const recallLabel = (item) => {
    if (item.kind !== 'user' && item.kind !== 'memory') return '과거 대화';
    const scope = item.kind === 'user' ? '항상 참고' : '관련 기억';
    return item.priority === 'preferred' ? `${scope} · 우선 적용` : `${scope} · 참고`;
  };
</script>

<nav class="context-rail" aria-label="컨텍스트 지도">
  <button class="context-rail-toggle" class:warning={percent >= 80} onclick={onToggle} title="컨텍스트 지도 · 입력 예산 대비 추정치">
    <span>{state?.input_budget > 0 ? `${percent}%` : '—'}</span>
  </button>
  <div class="context-marks" aria-hidden="true">
    {#each state?.segments || [] as segment}
      <button class="context-mark summarized" style:opacity={segment.id === state?.applied_segment_id ? 1 : 0.35} onclick={() => onJump(segment.start_message_id)} title={`요약 구간 ${segment.start_message_id}–${segment.end_message_id}`}></button>
    {/each}
    {#if state?.active_start_message_id}
      <button class="context-mark active" onclick={() => onJump(state.active_start_message_id)} title="현재 원문 컨텍스트"></button>
    {/if}
  </div>
</nav>

{#if open}
  <button class="context-backdrop" aria-label="컨텍스트 지도 닫기" onclick={onToggle}></button>
  <aside class="context-panel">
    <div class="context-panel-title"><div><strong>컨텍스트 지도</strong><small>{state?.preview ? '다음 요청 예상 · 저장된 첨부 정보 기준' : '현재 요청 · 전송 입력 기준'}</small></div><button onclick={onToggle}>×</button></div>
    {#if state}
      <div class="context-meter"><span style:width={`${Math.min(100, percent)}%`}></span></div>
      <div class="context-stats">
        <span>입력 추정 <strong>{state.estimated_tokens?.toLocaleString() || 0}</strong></span>
        <span>입력 예산 <strong>{state.input_budget?.toLocaleString() || '자동 감지 안 됨'}</strong></span>
        <span>원문·진행 중 대화 <strong>{state.active_tokens?.toLocaleString() || 0}</strong></span>
        <span>요약 <strong>{state.summary_tokens?.toLocaleString() || 0}</strong></span>
        <span>회수 <strong>{state.recall_tokens?.toLocaleString() || 0}</strong></span>
        <span>도구 결과 <strong>{state.tool_result_tokens?.toLocaleString() || 0}</strong></span>
        <span>시스템·도구 정의 <strong>{state.system_tool_tokens?.toLocaleString() || 0}</strong></span>
        {#if state.actual_tokens || state.last_request_tokens}<span>서버 실측 · 마지막 호출 <strong>{(state.actual_tokens || state.last_request_tokens).toLocaleString()}</strong></span>{/if}
      </div>
      {#if !state.managed}<p class="context-notice">{state.enabled ? '문맥 크기를 확인하지 못해 자동 관리가 적용되지 않습니다.' : '자동 관리 꺼짐 · 요약을 적용하지 않고 원문을 전달합니다.'}</p>{/if}
      {#if state.incomplete}<p class="context-notice">아직 변환되지 않은 첨부가 있어 추정치가 미완료입니다. 전송 전 다시 계산합니다.</p>{/if}
      {#if state.trimmed_tools}<p class="context-notice">오래된 도구 결과 {state.trimmed_tools}건 축소 · 원본은 보관되며 다시 읽을 수 있습니다.</p>{/if}
      {#if state.notice}<p class="context-notice">{state.notice}</p>{/if}
      <div class="context-legend"><span><i class="summarized"></i>현재 요약 · 흐린 표식은 이력</span><span><i class="active"></i>현재 원문</span></div>
      {#if state.recalls?.length}
        <section class="context-recalls">
          <strong>{state.preview ? '다음 요청에 참고할 기억·원문' : '이번 요청에 회수된 기억·원문'}</strong>
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
            <summary><span>{segment.id === state.applied_segment_id ? '현재 적용 요약' : `이전 요약 ${index + 1} · 미적용`}</span><small>추가 압축: 메시지 {segment.start_message_id}–{segment.end_message_id} · 원문 약 {segment.estimated_tokens?.toLocaleString()}토큰</small></summary>
            <small>이전 내용을 포함한 누적 요약 · 메시지 {segment.end_message_id}까지</small>
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
        <button class="primary" onclick={onCompact} disabled={disabled || loading || !state.managed}>{loading ? '정리 중…' : '지금 구간 정리'}</button>
      </div>
    {:else}<p class="context-notice">컨텍스트 정보를 불러오는 중입니다.</p>{/if}
  </aside>
{/if}
