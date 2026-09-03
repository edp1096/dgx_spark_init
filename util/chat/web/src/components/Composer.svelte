<script>
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
		<button class="attach" onclick={() => attachmentInput?.click()} disabled={!activeId || running || uploadingAttachments || voiceState !== 'idle' || pendingAttachments.length >= 6} aria-label="파일 첨부" title="이미지·음성·비디오·문서 첨부">＋</button>
      <button class="attach attach-url" onclick={() => sourceOpen = !sourceOpen} disabled={!activeId || running || uploadingAttachments || voiceState !== 'idle' || pendingAttachments.length >= 6} aria-label="URL 미디어 첨부" title="YouTube 등 URL에서 미디어 가져오기">⌁</button>
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
    </div>
    <textarea bind:this={element} bind:value={input} oninput={resizeComposerInput} onkeydown={onKeydown} onpaste={onPaste} placeholder={activeId ? '메시지를 입력하세요' : '새 대화를 만든 뒤 메시지를 입력하세요'} rows="1" disabled={!activeId || running}></textarea>
    <div class="composer-submit">
      {#if running}<button class="send stop" onclick={onStop} aria-label="응답 중지" title="응답 중지">■</button>{:else}<button class="send" onclick={onSend} disabled={!activeId || !input.trim() || uploadingAttachments || voiceState !== 'idle'} aria-label="메시지 전송" title="메시지 전송">↑</button>{/if}
    </div>
  </div>
	<small class:voice-active={voiceState !== 'idle'}>{voiceState === 'recording' ? `녹음 중 ${voiceDuration(voiceSeconds)} · 마이크를 다시 누르면 인식합니다` : voiceState === 'requesting' ? '마이크 연결 중…' : voiceState === 'transcribing' ? '음성 인식 중…' : `파일·URL 영상 첨부 · Enter 전송 · Shift+Enter 줄바꿈 · reasoning: ${reasoningEffort || '서버 기본값'} · 웹: ${webToolsEnabled ? '자동' : '꺼짐'}`}</small>
</footer>
