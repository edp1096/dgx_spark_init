<script>
  import MicrophoneHelp from './MicrophoneHelp.svelte';
  export let onManageWorkflows = () => {};
  import { listSkills } from '../api.js';
  import { tick } from 'svelte';
  import { attachmentAccept, attachmentKind, canPreviewVideo, formatAttachmentSize } from '../lib/attachments.js';

  export let pendingAttachments = [];
  export let uploadingAttachments = false;
  export let sourceDownloading = false;
  export let running = false;
  export let activeId = '';
  export let input = '';
  export let element;
  export let attachmentInput;
  export let reasoningEffort = '';
  export let webToolsEnabled = false;
  export let microphoneAvailable = false;
  export let voiceState = 'idle';
  export let voiceSeconds = 0;
  export let continuousVoiceEnabled = false;
  export let onRemoveAttachment = () => {};
  export let onAttachmentInputChange = () => {};
  export let onAttachURL = async () => false;
  export let onKeydown = () => {};
  export let onPaste = () => {};
  export let onStop = () => {};
  export let onSend = () => {};
  export let onStartVoice = () => {};
  export let onStopVoice = () => {};

  $: selectedSkills = input.match(/^(?:@skill:[a-z0-9-]+\s*)+/)?.[0].match(/@skill:[a-z0-9-]+/g) || [];
  function unselectSkill(marker) { input = input.replace(marker,'').trimStart(); }
  let workflowItems = [], previewWorkflow = null;
  $: selectedWorkflow = input.match(/^@(workflow|resume):([a-z0-9-]+)/)?.[0] || '';
  let skillOpen = false, skillItems = [], skillError = '', skillLoading = false;
  async function openSkills() {
    skillOpen = !skillOpen;
    if (!skillOpen) return;
    skillLoading = true; skillError = '';
    try { const response = await fetch('/api/workflows?selection=1'); if (!response.ok) throw new Error(await response.text()); workflowItems = await response.json(); skillItems = await listSkills(); } catch(error) { skillError = error.message; }
    finally { skillLoading = false; }
  }
  function chooseSkill(name) {
    const marker = `@skill:${name}`;
    input = input.replace(/^@(workflow|resume):[a-z0-9-]+\s*/, '');
    if (input.split(/\s+/).includes(marker)) unselectSkill(marker);
    else input = `${input.match(/^(?:@skill:[a-z0-9-]+\s*)*/)?.[0] || ''}${marker}\n${input.replace(/^(?:@skill:[a-z0-9-]+\s*)*/, '')}`;
    element?.focus();
  }
  function chooseWorkflow(item) {
    input = `@workflow:${item.name}\n${input.replace(/^(?:@(skill|workflow|resume):[a-z0-9-]+\s*)+/, '')}`;
    previewWorkflow=null;skillOpen=false;element?.focus();
  }
  function missingWorkflow(item) {
    return [...new Set([...(item.missing||[]),...item.steps.flatMap(step=>step.skills).filter(name=>{const s=skillItems.find(x=>x.name===name);return !s?.available || (s.toolsets.includes('web')&&!webToolsEnabled);})])];
  }
  let toolMenu, toolMenuButton, toolMenuOpen = false, menuLeft = 0, menuBottom = 0;
  function toggleToolMenu() {
    const rect = toolMenuButton.getBoundingClientRect();
    menuLeft = Math.max(8, Math.min(rect.left, window.innerWidth - 228));
    menuBottom = window.innerHeight - rect.top + 8;
    toolMenu.togglePopover();
  }
  function closeToolMenu() { toolMenu?.hidePopover(); }
  $: if (running || !activeId) { closeToolMenu(); skillOpen = false; }
  let sourceOpen = false;
  let sourceURL = '';
  let composerExpanded = false;

  async function resizeComposerInput() {
	await tick();
	if (!element) return;
	element.style.height = 'auto';
	const naturalHeight = element.scrollHeight;
	composerExpanded = input.includes('\n') || naturalHeight > 38 || (composerExpanded && input.length > 45);
	element.style.height = `${Math.min(Math.max(naturalHeight, 28), 180)}px`;
	element.style.overflowY = naturalHeight > 180 ? 'auto' : 'hidden';
  }

  $: {
	input;
	resizeComposerInput();
  }

  async function attachSource() {
	if (!sourceURL.trim() || uploadingAttachments) return;
	if (await onAttachURL(sourceURL.trim())) {
		sourceURL = '';
		sourceOpen = false;
	}
  }

  function sourceKeydown(event) {
	if (event.key === 'Enter') {
		event.preventDefault();
		attachSource();
	}
	if (event.key === 'Escape') sourceOpen = false;
  }

  function voiceDuration(seconds) {
	return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
  }
</script>

<svelte:window onresize={resizeComposerInput} />

<footer class="composer-footer">
  {#if skillOpen}<div class="composer-skills">
    <strong>스킬·작업 절차</strong><small>지정하지 않으면 필요한 스킬이나 작업 절차를 모델이 선택합니다.</small>
    <button onclick={()=>{input=input.replace(/^(?:@(skill|workflow|resume):[a-z0-9-]+\s*)+/, '');previewWorkflow=null;}}>자동 선택</button>
    <details><summary>작업 절차 선택 · 실행 순서 미리보기</summary>
    {#each workflowItems.filter(x=>x.enabled) as item}<button onclick={()=>previewWorkflow=item}>{item.name}<small>{item.description}</small></button>{/each}
    </details>
    {#if previewWorkflow}<section><strong>{previewWorkflow.name}</strong><ol>{#each previewWorkflow.steps as step}<li>{step.name} · {step.skills.join(', ')}<small>{step.done_when}</small></li>{/each}</ol>
    {#if missingWorkflow(previewWorkflow).length}<small>필요한 스킬·도구: {missingWorkflow(previewWorkflow).join(', ')}</small>{/if}
    <button disabled={missingWorkflow(previewWorkflow).length>0} onclick={()=>chooseWorkflow(previewWorkflow)}>이 순서로 적용</button><button onclick={()=>{skillOpen=false;onManageWorkflows();}}>라이브러리에서 편집</button></section>{/if}
    <small>개별 스킬은 여러 개 선택할 수 있습니다.</small>
    {#if skillLoading}<p>불러오는 중…</p>{/if}
    {#if skillError}<p role="alert">{skillError}</p>{/if}
    {#each skillItems as skill}<button disabled={!skill.available || (skill.toolsets.includes('web') && !webToolsEnabled)} onclick={() => chooseSkill(skill.name)} aria-pressed={selectedSkills.includes(`@skill:${skill.name}`)} title={skill.description}>{selectedSkills.includes(`@skill:${skill.name}`)?'✓ ':''}{skill.name}<small>{!skill.available ? skill.reason : skill.toolsets.includes('web') && !webToolsEnabled ? '웹 사용을 켜세요' : skill.description}</small></button>{/each}
    <button onclick={()=>skillOpen=false}>선택 닫기</button>
  </div>{/if}
  {#if selectedWorkflow || selectedSkills.length}<div class="composer-skill-control">{#if selectedWorkflow}<button onclick={()=>{input=input.replace(/^@(workflow|resume):[a-z0-9-]+\s*/, '');}} disabled={running}>{selectedWorkflow.startsWith('@resume:')?'작업 이어하기':selectedWorkflow.replace(/^@workflow:/,'')} ×</button>{/if}{#each selectedSkills as marker}<button onclick={() => unselectSkill(marker)} disabled={running} aria-label={`${marker.slice(7)} 지정 해제`}>{marker.slice(7)} ×</button>{/each}</div>{/if}
  {#if pendingAttachments.length || uploadingAttachments}
    <div class="pending-attachments">
      {#each pendingAttachments as attachment}
        <div class="pending-attachment">
          {#if attachmentKind(attachment) === 'image'}
            <img src={attachment.url} alt={attachment.name} />
          {:else if attachmentKind(attachment) === 'video' && canPreviewVideo(attachment)}
            <video src={attachment.url} muted preload="metadata" aria-label={attachment.name}></video>
          {:else}
			<span class="media-file-icon">{attachmentKind(attachment) === 'audio' ? '♪' : attachmentKind(attachment) === 'document' ? '▤' : '▶'}</span>
          {/if}
          <span class="pending-media-name" title={attachment.name}>{attachment.name}<small>{formatAttachmentSize(attachment.size)}</small></span>
          <button onclick={() => onRemoveAttachment(attachment.id)} disabled={running} aria-label={`${attachment.name} 첨부 제거`}>×</button>
        </div>
      {/each}
      {#if uploadingAttachments}<span class="uploading">{sourceDownloading ? '480~720p URL 영상 취득 중…' : '미디어 업로드 중…'}</span>{/if}
    </div>
  {/if}
  {#if sourceOpen}
    <div class="media-url-row">
      <input bind:value={sourceURL} onkeydown={sourceKeydown} placeholder="YouTube · Vimeo · Dailymotion 등 영상 주소" disabled={uploadingAttachments || running} aria-label="미디어 주소" />
      <button onclick={attachSource} disabled={!sourceURL.trim() || uploadingAttachments || running}>{sourceDownloading ? '취득 중…' : '첨부'}</button>
      <button class="url-close" onclick={() => sourceOpen = false} disabled={uploadingAttachments} aria-label="주소 입력 닫기">×</button>
    </div>
  {/if}
  <div class="composer" class:composer-expanded={composerExpanded} role="group" aria-label="메시지와 미디어 입력">
    <input class="media-input" bind:this={attachmentInput} type="file" accept={attachmentAccept} multiple onchange={onAttachmentInputChange} />
    <div class="composer-tools">
      <button class="attach" bind:this={toolMenuButton} onclick={toggleToolMenu} disabled={!activeId || running} aria-label="입력 도구 열기" aria-expanded={toolMenuOpen} aria-controls="composer-tool-menu" title="첨부·스킬·작업 절차">＋</button>
      <div id="composer-tool-menu" class="composer-tool-menu" popover="auto" bind:this={toolMenu} ontoggle={event=>toolMenuOpen=event.newState==='open'} style:left={`${menuLeft}px`} style:bottom={`${menuBottom}px`}>
        <button onclick={()=>{closeToolMenu();attachmentInput?.click();}} disabled={uploadingAttachments || voiceState !== 'idle' || pendingAttachments.length >= 6} aria-label="파일 첨부">파일 첨부</button>
        <button onclick={()=>{closeToolMenu();sourceOpen=!sourceOpen;}} disabled={uploadingAttachments || voiceState !== 'idle' || pendingAttachments.length >= 6} aria-label="URL 미디어 첨부">URL 미디어 첨부</button>
        <button onclick={()=>{closeToolMenu();openSkills();}} aria-expanded={skillOpen}>스킬·작업 절차</button>
      </div>
      <button
        class="attach voice-input"
        class:recording={voiceState === 'recording'}
        class:processing={voiceState === 'requesting' || voiceState === 'transcribing'}
        onclick={voiceState === 'recording' ? onStopVoice : onStartVoice}
        disabled={!activeId || running || uploadingAttachments || !microphoneAvailable || continuousVoiceEnabled || (voiceState !== 'idle' && voiceState !== 'recording')}
        aria-label={voiceState === 'recording' ? '녹음 정지 후 음성 인식' : '음성으로 입력'}
        title={!microphoneAvailable ? 'HTTPS 또는 안전한 출처 설정이 필요합니다' : continuousVoiceEnabled ? '연속 음성 모드를 끈 뒤 사용할 수 있습니다' : voiceState === 'recording' ? '녹음 정지 후 음성 인식' : '음성으로 입력'}
      >
        {#if voiceState === 'transcribing' || voiceState === 'requesting'}<span class="voice-spinner"></span>{:else}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Zm-6 9a6 6 0 0 0 12 0M12 18v3m-3 0h6" /></svg>{/if}
      </button>
      {#if !microphoneAvailable}<MicrophoneHelp compact />{/if}
    </div>
    <textarea bind:this={element} bind:value={input} oninput={resizeComposerInput} onkeydown={onKeydown} onpaste={onPaste} placeholder={activeId ? '메시지를 입력하세요' : '새 대화를 만든 뒤 메시지를 입력하세요'} rows="1" disabled={!activeId || running}></textarea>
    <div class="composer-submit">
      {#if running}<button class="send stop" onclick={onStop} aria-label="응답 중지" title="응답 중지">■</button>{:else}<button class="send" onclick={onSend} disabled={!activeId || !input.trim() || uploadingAttachments || voiceState !== 'idle'} aria-label="메시지 전송" title="메시지 전송">↑</button>{/if}
    </div>
  </div>
	<small class:voice-active={voiceState !== 'idle'}>{voiceState === 'recording' ? `녹음 중 ${voiceDuration(voiceSeconds)} · 마이크를 다시 누르면 인식합니다` : voiceState === 'requesting' ? '마이크 연결 중…' : voiceState === 'transcribing' ? '음성 인식 중…' : `파일·URL 영상 첨부 · Enter 전송 · Shift+Enter 줄바꿈 · reasoning: ${reasoningEffort || '서버 기본값'} · 웹: ${webToolsEnabled ? '자동' : '꺼짐'}`}</small>
</footer>

<style>
.composer-tool-menu{position:fixed;top:auto;right:auto;margin:0;width:220px;box-sizing:border-box;padding:6px;border:1px solid #303744;border-radius:12px;background:#191e28;color:#d7dfef;box-shadow:0 8px 28px #0004}
.composer-tool-menu:popover-open{display:grid;gap:3px}
.composer-tool-menu button{text-align:left;padding:10px;border:0;border-radius:7px;background:transparent;color:inherit;font-size:13px;cursor:pointer}
.composer-tool-menu button:hover{background:#8994a422}
.composer-tool-menu button:disabled{opacity:.45;cursor:default}
:global(html[data-theme="light"]) .composer-tool-menu{background:#fff;color:#42516b;border-color:#d8dee8}

.composer-skills{display:grid;gap:8px;max-height:280px;overflow:auto;padding:12px;border:1px solid var(--border,#555);border-radius:10px}.composer-skills button{text-align:left;background:#191e28;color:#d7dfef;border:1px solid #303744;border-radius:7px;padding:8px}.composer-skills button:disabled{opacity:.45;cursor:default}.composer-skills small{display:block;opacity:.7}.composer-skill-control{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:6px}.composer-skill-control button{font-size:12px;padding:4px 10px;border:1px solid #303744;border-radius:7px;background:#191e28;color:#a9b6cc}.composer-skill-control small{font-size:11px}
:global(html[data-theme="light"]) .composer-skills button,:global(html[data-theme="light"]) .composer-skill-control button{background:#f3f5f8;color:#42516b;border-color:#d8dee8}
</style>
