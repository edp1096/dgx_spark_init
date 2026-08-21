<script>
  export let activeSession = null;
  export let running = false;
  export let editingTitle = false;
  export let titleInput = '';
  export let titleEditor;
  export let models = [];
  export let selectedModel = '';
  export let reasoningEffort = '';
  export let webToolsEnabled = false;
  export let health = { status: 'checking', model: '' };
  export let sshGrants = [];
  export let controlsOpen = false;
  export let onToggleSidebar = () => {};
  export let onBeginTitleEdit = () => {};
  export let onTitleKeydown = () => {};
  export let onSaveTitle = () => {};
  export let onToggleControls = () => {};
  export let onCloseControls = () => {};
  export let onRevokeSSHGrant = () => {};
  export let onClearSSHGrants = () => {};
</script>

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
    <input bind:value={reasoningEffort} list="reasoning-levels" placeholder="reasoning effort" aria-label="Reasoning effort" />
    <button class:active={webToolsEnabled} class="web-toggle" onclick={() => webToolsEnabled = !webToolsEnabled} title="모델이 필요할 때 웹검색 사용">{webToolsEnabled ? '웹검색 자동' : '웹검색 꺼짐'}</button>
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
      <option value="none"></option><option value="minimal"></option><option value="low"></option>
      <option value="medium"></option><option value="high"></option><option value="xhigh"></option><option value="max"></option>
    </datalist>
    <span class:offline={health.status !== 'ok'} class="status">● {health.status === 'ok' ? '연결됨' : '연결 오류'}</span>
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
    <label>Reasoning effort<input bind:value={reasoningEffort} list="reasoning-levels" placeholder="reasoning effort" /></label>
    <button class:active={webToolsEnabled} class="drawer-web-toggle" onclick={() => webToolsEnabled = !webToolsEnabled}>{webToolsEnabled ? '웹검색 자동' : '웹검색 꺼짐'}</button>
    {#if sshGrants.length}
      <section class="drawer-ssh-grants">
        <div><strong>SSH 자동 허용</strong><button onclick={onClearSSHGrants}>모두 해제</button></div>
        {#each sshGrants as grant}
          <p><span><b>{grant.host_name}</b><small>{grant.host_alias}</small></span><button onclick={() => onRevokeSSHGrant(grant.host_id)}>해제</button></p>
        {/each}
      </section>
    {/if}
    <div class="drawer-status"><span class:offline={health.status !== 'ok'}>● {health.status === 'ok' ? '연결됨' : '연결 오류'}</span><small>{selectedModel || health.model || '모델 확인 중'}</small></div>
  </div>
{/if}
