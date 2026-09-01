<script>
  import { modelCapabilities, normalizeReasoningEffort, thinkingToggleValue } from '../lib/model-capabilities.js';
  import ReasoningEffortControl from './ReasoningEffortControl.svelte';
  import RuntimePanel from './RuntimePanel.svelte';

  export let activeSession = null;
  export let running = false;
  export let editingTitle = false;
  export let titleInput = '';
  export let titleEditor;
  export let models = [];
  export let selectedModel = '';
  export let modelType = 'generic';
  export let reasoningEffort = '';
  export let webToolsEnabled = false;
  export let health = { status: 'checking', model: '' };
  export let runtime = null;
  export let runtimeBusy = false;
  export let sshGrants = [];
  export let microphoneAvailable = false;
  export let continuousVoiceEnabled = false;
  export let continuousVoiceState = 'off';
  export let continuousQueueCount = 0;
  export let controlsOpen = false;
  export let onToggleSidebar = () => {};
  export let onBeginTitleEdit = () => {};
  export let onTitleKeydown = () => {};
  export let onSaveTitle = () => {};
  export let onToggleControls = () => {};
  export let onCloseControls = () => {};
  export let onRevokeSSHGrant = () => {};
  export let onClearSSHGrants = () => {};
  export let onToggleContinuousVoice = () => {};
  export let onRefreshHealth = () => {};
  export let onRuntimeAction = async () => {};
  export let onRefreshRuntime = async () => {};

  let statusOpen = false;

  $: voiceModeActive = continuousVoiceEnabled || !['off', 'error', 'stopping'].includes(continuousVoiceState);
  $: runtimeStarting = runtime?.operation?.state === 'running';
  $: modelProfile = modelCapabilities(modelType);
  $: gemmaThinkingValue = thinkingToggleValue(reasoningEffort);
  $: if (modelProfile.family === 'qwen3.8') reasoningEffort = normalizeReasoningEffort(modelType, reasoningEffort);

  function toggleThinking() {
    reasoningEffort = gemmaThinkingValue === 'on' ? 'none' : 'on';
  }
  $: voiceModeLabel = continuousVoiceState === 'requesting' ? '마이크 연결 중'
    : continuousVoiceState === 'stopping' ? '음성대기 종료 중'
    : continuousVoiceState === 'speaking' ? '발화 감지'
    : continuousVoiceState === 'paused' ? 'AI 음성 재생 중'
    : continuousQueueCount ? `음성인식 ${continuousQueueCount}`
    : continuousVoiceState === 'calibrating' ? '소음 측정 중'
    : voiceModeActive ? '음성대기 켜짐' : '음성대기 꺼짐';
  $: serviceRows = [
    { label: '모델 API', status: health.status, detail: selectedModel || health.model || health.endpoint || '' },
    { label: '미디어 처리', status: health.asr?.ffmpeg?.status || 'disabled', detail: 'SparkTalk Extra' },
    { label: 'ASR API', status: health.asr?.asr?.status || 'disabled', detail: '마이크·미디어 음성 인식' },
    { label: 'TTS API', status: health.tts?.status || 'disabled', detail: health.tts?.model || '음성 읽기' },
    { label: '이미지 API', status: health.image?.status || 'disabled', detail: health.image?.model || '이미지 도구' },
    { label: 'Extra SSH', status: health.extra?.ssh?.status || 'disabled', detail: 'SSH 도구' },
  ];

  function statusLabel(status) {
    if (status === 'ok') return '온라인';
    if (status === 'disabled') return '사용 안 함';
    if (status === 'checking') return '확인 중';
    return '오프라인';
  }

  function toggleStatus(event) {
    event.stopPropagation();
    statusOpen = !statusOpen;
    if (statusOpen) onRefreshHealth();
  }

  function refreshStatus(event) {
    event.stopPropagation();
    onRefreshHealth();
  }

  function closeStatusOutside(event) {
    if (!event.target.closest('.connection-menu, .drawer-status')) statusOpen = false;
  }
</script>

<svelte:window onclick={closeStatusOutside} />

<header>
  <button class="sidebar-toggle" onclick={onToggleSidebar} aria-label="사이드바 열기 또는 닫기">☰</button>
  <div class="chat-heading">
    {#if editingTitle}
      <input class="title-editor" bind:this={titleEditor} bind:value={titleInput} maxlength="120" onkeydown={onTitleKeydown} onblur={onSaveTitle} aria-label="대화 제목" />
    {:else}
      <button class="chat-title" onclick={onBeginTitleEdit} disabled={!activeSession || running} title="대화 제목 수정"><span>{activeSession?.title || '새 대화'}</span><i>✎</i></button>
    {/if}
  </div>
  <div class="model-controls">
    <select bind:value={selectedModel} aria-label="모델 선택">
      {#if !models.length}<option value={selectedModel}>{selectedModel || '모델 없음'}</option>{/if}
      {#each models as model}<option value={model}>{model}</option>{/each}
    </select>
    {#if modelProfile.reasoning === 'toggle'}
      <button class="thinking-toggle" class:active={gemmaThinkingValue === 'on'} onclick={toggleThinking} aria-pressed={gemmaThinkingValue === 'on'}>{gemmaThinkingValue === 'on' ? 'Thinking 켜짐' : 'Thinking 꺼짐'}</button>
    {:else if modelProfile.family === 'qwen3.8'}
      <ReasoningEffortControl bind:value={reasoningEffort} />
    {:else}
      <input bind:value={reasoningEffort} list="reasoning-levels" placeholder="reasoning effort" aria-label="Reasoning effort" />
    {/if}
    <button class:active={webToolsEnabled} class="web-toggle" onclick={() => webToolsEnabled = !webToolsEnabled} title="모델이 필요할 때 웹검색 사용">{webToolsEnabled ? '웹검색 자동' : '웹검색 꺼짐'}</button>
    <button class:active={voiceModeActive} class:speaking={continuousVoiceState === 'speaking'} class="voice-mode-toggle" onclick={onToggleContinuousVoice} disabled={!activeSession || !microphoneAvailable || ['requesting', 'stopping'].includes(continuousVoiceState)} aria-pressed={voiceModeActive} title={!microphoneAvailable ? 'HTTPS 또는 안전한 출처 설정이 필요합니다' : '발화를 자동 인식하고 바로 전송'}>{voiceModeLabel}</button>
    {#if sshGrants.length}
      <details class="ssh-grants-menu">
        <summary title="이 대화에서 자동 허용된 SSH 서버">SSH 허용 {sshGrants.length}</summary>
        <div class="ssh-grants-popover">
          <div class="ssh-grants-heading"><strong>이 대화에서 허용</strong><button onclick={onClearSSHGrants}>모두 해제</button></div>
          {#each sshGrants as grant}
            <div class="ssh-grant-row"><span><b>{grant.host_name}</b><small>{grant.host_alias}</small></span><button onclick={() => onRevokeSSHGrant(grant.host_id)}>해제</button></div>
          {/each}
        </div>
      </details>
    {/if}
    <datalist id="reasoning-levels">
      {#each modelProfile.reasoningLevels as level}<option value={level}></option>{/each}
    </datalist>
    <div class="connection-menu">
      <button class:starting={runtimeStarting} class:offline={!runtimeStarting && health.status !== 'ok'} class="status" onclick={toggleStatus} aria-expanded={statusOpen}>● {runtimeStarting ? '기동 중' : health.status === 'ok' ? '연결됨' : '연결 오류'}</button>
      {#if statusOpen}
        <div class="connection-popover" role="dialog" aria-label="DGX Spark 운영 상태" tabindex="-1">
          {#if runtime}
            <RuntimePanel {runtime} busy={runtimeBusy} onAction={onRuntimeAction} onRefresh={onRefreshRuntime} />
          {:else}
            <div class="connection-heading"><strong>연결 상태</strong><button onclick={refreshStatus} title="상태 새로고침" aria-label="상태 새로고침">↻</button></div>
            {#each serviceRows as service}
              <div class="connection-row">
                <span class:online={service.status === 'ok'} class:inactive={service.status === 'disabled'}><i></i>{service.label}</span>
                <div><b>{statusLabel(service.status)}</b><small title={service.detail}>{service.detail}</small></div>
              </div>
            {/each}
          {/if}
        </div>
      {/if}
    </div>
  </div>
  <button class="mobile-controls-toggle" class:active={controlsOpen} onclick={onToggleControls} aria-label="모델 및 대화 설정" aria-expanded={controlsOpen}>☷</button>
</header>
{#if controlsOpen}
  <button class="controls-backdrop" onclick={onCloseControls} aria-label="모델 설정 패널 닫기"></button>
  <div class="controls-drawer" role="dialog" aria-modal="true" aria-label="모델 및 대화 설정">
    <div class="controls-title"><strong>대화 설정</strong><button onclick={onCloseControls} aria-label="닫기">×</button></div>
    <label>모델
      <select bind:value={selectedModel} aria-label="모델 선택">
        {#if !models.length}<option value={selectedModel}>{selectedModel || '모델 없음'}</option>{/if}
        {#each models as model}<option value={model}>{model}</option>{/each}
      </select>
    </label>
    {#if modelProfile.reasoning === 'toggle'}
      <button class="drawer-thinking-toggle" class:active={gemmaThinkingValue === 'on'} onclick={toggleThinking} aria-pressed={gemmaThinkingValue === 'on'}>{gemmaThinkingValue === 'on' ? 'Thinking 켜짐' : 'Thinking 꺼짐'}</button>
    {:else if modelProfile.family === 'qwen3.8'}
      <ReasoningEffortControl bind:value={reasoningEffort} drawer={true} />
    {:else}
      <label>Reasoning effort<input bind:value={reasoningEffort} list="reasoning-levels" placeholder="reasoning effort" /></label>
    {/if}
    <button class:active={webToolsEnabled} class="drawer-web-toggle" onclick={() => webToolsEnabled = !webToolsEnabled}>{webToolsEnabled ? '웹검색 자동' : '웹검색 꺼짐'}</button>
    <button class:active={voiceModeActive} class:speaking={continuousVoiceState === 'speaking'} class="drawer-voice-toggle" onclick={onToggleContinuousVoice} disabled={!activeSession || !microphoneAvailable || ['requesting', 'stopping'].includes(continuousVoiceState)} aria-pressed={voiceModeActive}>{voiceModeLabel}</button>
    {#if sshGrants.length}
      <section class="drawer-ssh-grants">
        <div><strong>SSH 자동 허용</strong><button onclick={onClearSSHGrants}>모두 해제</button></div>
        {#each sshGrants as grant}
          <p><span><b>{grant.host_name}</b><small>{grant.host_alias}</small></span><button onclick={() => onRevokeSSHGrant(grant.host_id)}>해제</button></p>
        {/each}
      </section>
    {/if}
    <div class="drawer-status">
      <button class:starting={runtimeStarting} class:offline={!runtimeStarting && health.status !== 'ok'} onclick={toggleStatus} aria-expanded={statusOpen}><span>● {runtimeStarting ? '기동 중' : health.status === 'ok' ? '연결됨' : '연결 오류'}</span><small>{selectedModel || health.model || '모델 확인 중'}</small></button>
      {#if statusOpen}
        <div class="drawer-services">
          {#if runtime}
            <RuntimePanel {runtime} busy={runtimeBusy} onAction={onRuntimeAction} onRefresh={onRefreshRuntime} />
          {:else}
            <div class="connection-heading"><strong>연결 상태</strong><button onclick={refreshStatus} title="상태 새로고침" aria-label="상태 새로고침">↻</button></div>
            {#each serviceRows as service}
              <div class="connection-row">
                <span class:online={service.status === 'ok'} class:inactive={service.status === 'disabled'}><i></i>{service.label}</span>
                <div><b>{statusLabel(service.status)}</b><small title={service.detail}>{service.detail}</small></div>
              </div>
            {/each}
          {/if}
        </div>
      {/if}
    </div>
  </div>
{/if}
