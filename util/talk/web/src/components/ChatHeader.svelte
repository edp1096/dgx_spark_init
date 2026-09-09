<script>
  import MicrophoneHelp from './MicrophoneHelp.svelte';
  import { tick } from 'svelte';
  import { modelCapabilities, normalizeReasoningEffort, thinkingToggleValue, reasoningEffortLabel } from '../lib/model-capabilities.js';
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
  let controlsSection = 'model';
  let modelTrigger, toolsTrigger, statusTrigger;
  $: reasoningSummary = modelProfile.reasoning === 'toggle' ? (gemmaThinkingValue === 'on' ? 'Thinking 켜짐' : 'Thinking 꺼짐') : reasoningEffortLabel(reasoningEffort, modelType);
  async function openControls(section) {
    statusOpen = false;
    if (controlsOpen && controlsSection === section) { onCloseControls(); return; }
    controlsSection = section;
    if (!controlsOpen) onToggleControls();
    await tick();
    document.querySelector('.quick-panel .quick-field select, .quick-panel .quick-field input, .quick-panel .quick-field button:not(:disabled)')?.focus();
  }
  function closeQuickPanel() { onCloseControls(); (controlsSection === 'model' ? modelTrigger : toolsTrigger)?.focus(); }
  function onEscape(event) {
    if (event.key !== 'Escape') return;
    if (controlsOpen) { event.preventDefault(); closeQuickPanel(); }
    if (statusOpen) { event.preventDefault(); statusOpen = false; statusTrigger?.focus(); }
  }

  $: voiceModeActive = continuousVoiceEnabled || !['off', 'error', 'stopping'].includes(continuousVoiceState);
  $: runtimeStarting = runtime?.operation?.state === 'running';
  $: modelProfile = modelCapabilities(modelType);
  $: gemmaThinkingValue = thinkingToggleValue(reasoningEffort);
  $: if (modelProfile.family === 'qwen3.8' || modelProfile.family === 'qwen3.8-exl3' || (modelProfile.family === 'glm5.3' || modelProfile.family === 'deepseek-v4')) reasoningEffort = normalizeReasoningEffort(modelType, reasoningEffort);

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
    { label: '문서 생성', status: health.extra?.documents?.status || 'disabled', detail: '문서·표·발표자료' },
    { label: '웹 자료 수집', status: health.extra?.collector?.status || 'disabled', detail: 'Collector' },
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
    onCloseControls();
    statusOpen = !statusOpen;
    if (statusOpen) onRefreshHealth();
  }

  function refreshStatus(event) {
    event.stopPropagation();
    onRefreshHealth();
  }

  function closeStatusOutside(event) {
    if (!event.target.closest?.('.connection-menu')) statusOpen = false;
    if (controlsOpen && !event.target.closest?.('.quick-panel, .model-menu-toggle, .tools-menu-toggle, .settings-help-dialog')) onCloseControls();
  }
</script>

<svelte:window onclick={closeStatusOutside} onkeydown={onEscape} />

<header class="chat-header">
  <button class="sidebar-toggle" onclick={onToggleSidebar} aria-label="사이드바 열기 또는 닫기">☰</button>
  <div class="chat-heading">
    {#if editingTitle}
      <input class="title-editor" bind:this={titleEditor} bind:value={titleInput} maxlength="120" onkeydown={onTitleKeydown} onblur={onSaveTitle} aria-label="대화 제목" />
    {:else}
      <button class="chat-title" onclick={onBeginTitleEdit} disabled={!activeSession || running} title="대화 제목 수정"><span>{activeSession?.title || '새 대화'}</span><i>✎</i></button>
    {/if}
  </div>
  <div class="header-actions">
    <button bind:this={modelTrigger} class="model-menu-toggle" class:active={controlsOpen && controlsSection === 'model'} onclick={() => openControls('model')} aria-label="모델 및 대화 설정" aria-haspopup="dialog" aria-expanded={controlsOpen && controlsSection === 'model'} title={`${selectedModel || '모델 없음'} · ${reasoningSummary || '추론 기본값'}`}>
      <span class="model-short-label">모델</span><span class="model-name">{selectedModel || '모델 없음'}</span><small class="reason-summary">{reasoningSummary}</small><span aria-hidden="true">⌄</span>
    </button>
    <button bind:this={toolsTrigger} class="tools-menu-toggle" class:active={controlsOpen && controlsSection === 'tools'} class:voice-active={voiceModeActive} class:speaking={continuousVoiceState === 'speaking'} onclick={() => openControls('tools')} aria-label="대화 도구" aria-haspopup="dialog" aria-expanded={controlsOpen && controlsSection === 'tools'} title={`${webToolsEnabled ? '웹검색 자동' : '웹검색 꺼짐'} · ${voiceModeLabel}${sshGrants.length ? ` · SSH 허용 ${sshGrants.length}` : ''}`}>
      도구{#if webToolsEnabled || voiceModeActive || sshGrants.length}<i class="tool-active-dot" aria-label="사용 중인 기능 있음"></i>{/if}<span aria-hidden="true">⌄</span>
    </button>
    <div class="connection-menu">
      <button bind:this={statusTrigger} class:starting={runtimeStarting} class:offline={!runtimeStarting && health.status !== 'ok'} class="status" onclick={toggleStatus} aria-expanded={statusOpen} aria-haspopup="dialog">● {runtimeStarting ? '기동 중' : health.status === 'ok' ? '연결됨' : '연결 오류'}</button>
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
</header>

{#if controlsOpen}
  <div class="quick-panel" role="dialog" aria-label={controlsSection === 'model' ? '모델 및 대화 설정' : '대화 도구 설정'}>
    <div class="quick-heading"><strong>{controlsSection === 'model' ? '모델·추론' : '대화 도구'}</strong><button type="button" onclick={closeQuickPanel} aria-label="대화 제어 닫기">×</button></div>
    {#if controlsSection === 'model'}
      <label class="quick-field"><span>모델</span><select bind:value={selectedModel} aria-label="모델 선택">
        {#if !models.length}<option value={selectedModel}>{selectedModel || '모델 없음'}</option>{/if}
        {#each models as model}<option value={model}>{model}</option>{/each}
      </select></label>
      {#if modelProfile.reasoning === 'toggle'}
        <div class="quick-field"><span>추론</span><button class="thinking-toggle" class:active={gemmaThinkingValue === 'on'} onclick={toggleThinking} aria-pressed={gemmaThinkingValue === 'on'}>{gemmaThinkingValue === 'on' ? 'Thinking 켜짐' : 'Thinking 꺼짐'}</button></div>
      {:else if modelProfile.family === 'qwen3.8' || modelProfile.family === 'qwen3.8-exl3' || modelProfile.family === 'glm5.3' || modelProfile.family === 'deepseek-v4'}
        <div class="quick-field"><span>추론 강도</span><ReasoningEffortControl bind:value={reasoningEffort} {modelType} /></div>
      {:else}
        <label class="quick-field"><span>추론 강도</span><input bind:value={reasoningEffort} list="reasoning-levels" placeholder="기본값" aria-label="Reasoning effort" /></label>
      {/if}
      <small class="quick-note">변경한 값은 다음 메시지부터 적용됩니다.</small>
    {:else}
      <div class="quick-field"><span>웹검색</span><button class="web-toggle" class:active={webToolsEnabled} onclick={() => webToolsEnabled = !webToolsEnabled} aria-label="웹검색 자동 사용" aria-pressed={webToolsEnabled}>{webToolsEnabled ? '자동' : '꺼짐'}</button></div>
      <div class="quick-field"><span>음성대기</span><button class="voice-mode-toggle" class:active={voiceModeActive} class:speaking={continuousVoiceState === 'speaking'} onclick={onToggleContinuousVoice} disabled={!activeSession || !microphoneAvailable || ['requesting', 'stopping'].includes(continuousVoiceState)} aria-pressed={voiceModeActive} aria-label={voiceModeLabel}>{voiceModeLabel.replace(/^음성대기 /, '')}</button></div>
      {#if !microphoneAvailable}<MicrophoneHelp />{:else if !activeSession}<small class="quick-note">새 대화를 만든 뒤 음성대기를 켤 수 있습니다.</small>{/if}
      {#if sshGrants.length}
        <section class="quick-ssh-grants">
          <div class="quick-heading"><strong>이 대화의 SSH 허용</strong><button class="danger" onclick={onClearSSHGrants}>모두 해제</button></div>
          {#each sshGrants as grant}<div class="quick-grant"><span><b>{grant.host_name}</b><small>{grant.host_alias}</small></span><button class="danger" onclick={() => onRevokeSSHGrant(grant.host_id)}>해제</button></div>{/each}
        </section>
      {/if}
      <button class="service-status-link" onclick={toggleStatus}>Extra·모델 서비스 상태 보기 →</button>
    {/if}
  </div>
{/if}
<datalist id="reasoning-levels">{#each modelProfile.reasoningLevels as level}<option value={level}></option>{/each}</datalist>

<style>
  .chat-header { gap: 10px; }
  .chat-header .chat-heading { min-width: 0; }
  .header-actions { display: flex; flex: 0 1 auto; align-items: center; gap: 7px; min-width: 0; }
  .header-actions, .quick-panel { --quick-bg: #171b23; --quick-input: #171a22; --quick-border: #343a48; --quick-text: #cbd1dc; --quick-muted: #8993a5; --quick-hover: #252d3b; }
  .model-menu-toggle, .tools-menu-toggle { display: flex; height: 34px; align-items: center; gap: 7px; min-width: 0; padding: 0 10px; border: 1px solid var(--quick-border); border-radius: 8px; background: var(--quick-input); color: var(--quick-text); font-size: 12px; }
  .model-menu-toggle:hover, .tools-menu-toggle:hover { background: var(--quick-hover); }
  .model-menu-toggle.active, .tools-menu-toggle.active { border-color: #6584ed; }
  .model-name { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .model-short-label { display: none; }
  .reason-summary { flex-shrink: 0; font-size: 10px; color: var(--quick-muted); }
  .tools-menu-toggle { flex-shrink: 0; }
  .tool-active-dot { width: 5px; height: 5px; border-radius: 50%; background: #6584ed; }
  .voice-active .tool-active-dot { background: #68c995; }
  .speaking .tool-active-dot { background: #ed727f; animation: voice-recording-pulse 1.2s ease-in-out infinite; }
  .chat-header .status { display: block; padding: 8px 0; }
  .chat-header .connection-popover { max-height: calc(100dvh - 80px); overflow-y: auto; overscroll-behavior: contain; touch-action: pan-y; }
  .quick-panel { position: fixed; z-index: 20; top: 64px; right: 20px; display: grid; gap: 14px; width: min(390px, calc(100vw - 24px)); max-height: calc(100dvh - 80px); overflow-y: auto; padding: 14px; border: 1px solid var(--quick-border); border-radius: 12px; color: var(--quick-text); background: var(--quick-bg); box-shadow: 0 12px 35px #0005; }
  .quick-heading { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
  .quick-heading strong { font-size: 13px; }
  .quick-panel button { padding: 7px 10px; border: 1px solid var(--quick-border); border-radius: 8px; color: var(--quick-text); background: var(--quick-input); font: inherit; font-size: 12px; }
  .quick-panel button:hover:not(:disabled) { background: var(--quick-hover); }
  .quick-heading > button[aria-label] { width: 28px; height: 28px; padding: 0; border: 0; font-size: 20px; }
  .quick-field { display: grid; grid-template-columns: 84px minmax(0, 1fr); align-items: center; gap: 10px; min-width: 0; font-size: 12px; }
  .quick-field select, .quick-field > input { width: 100%; min-width: 0; height: 34px; padding: 0 8px; border: 1px solid var(--quick-border); border-radius: 8px; background: var(--quick-input); color: var(--quick-text); font: inherit; }
  .quick-panel button.active { border-color: #6584ed; color: #83a6f2; background: #6584ed15; }
  .quick-panel button.speaking { border-color: #ed727f; color: #ed727f; }
  .quick-panel button:disabled { opacity: .45; cursor: default; }
  .quick-panel button.danger { color: #d66d79; }
  .quick-note { color: var(--quick-muted); font-size: 11px; line-height: 1.5; }
  .quick-ssh-grants { display: grid; gap: 8px; border-top: 1px solid var(--quick-border); padding-top: 10px; }
  .quick-grant { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .quick-grant > span { display: grid; gap: 3px; min-width: 0; }
  .quick-grant b { font-size: 12px; overflow-wrap: anywhere; }
  .quick-grant small { font-size: 10px; color: var(--quick-muted); }
  .quick-panel .service-status-link { border: 0; background: transparent; text-align: left; padding: 4px 0; color: var(--quick-muted); }
  .quick-panel :global(.qwen-effort-control) { width: 100%; border-color: var(--quick-border); color: var(--quick-text); background: var(--quick-input); }
  .quick-panel :global(.qwen-effort-control input) { flex: 1; width: 100%; min-width: 0; height: 18px; accent-color: #6584ed; }
  .quick-panel :global(.qwen-effort-control output) { color: var(--quick-text); }
  button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid #6584ed; outline-offset: 2px; }
  :global(html[data-theme="light"]) .header-actions, :global(html[data-theme="light"]) .quick-panel { --quick-bg: #fff; --quick-input: #f3f5f8; --quick-border: #ccd4e0; --quick-text: #344054; --quick-muted: #68758a; --quick-hover: #e8edf5; }
  @media (max-width: 900px) { .model-name { max-width: 135px; } .reason-summary { display: none; } }
  @media (max-width: 600px) {
    .chat-header { gap: 6px; padding: 10px 8px; }
    .header-actions { gap: 5px; }
    .model-name, .reason-summary { display: none; }
    .model-short-label { display: inline; }
    .model-menu-toggle, .tools-menu-toggle { padding: 0 7px; gap: 4px; font-size: 11px; }
    .chat-header .status { font-size: 10px; }
    .chat-header .chat-title { padding: 0 3px; font-size: 12px; }
    .quick-panel { top: 58px; right: 8px; width: calc(100vw - 16px); max-height: calc(100dvh - 74px); }
    .chat-header .connection-popover { position: fixed; top: 58px; right: 8px; width: calc(100vw - 16px); max-height: calc(100dvh - 74px); }
  }
</style>
