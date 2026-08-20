<script>
  import { attachmentAccept, attachmentKind, canPreviewVideo, formatAttachmentSize } from '../lib/attachments.js';

  export let pendingAttachments = [];
  export let uploadingAttachments = false;
  export let sourceDownloading = false;
  export let running = false;
  export let activeId = '';
  export let input = '';
  export let attachmentInput;
  export let reasoningEffort = '';
  export let webToolsEnabled = false;
  export let onRemoveAttachment = () => {};
  export let onAttachmentInputChange = () => {};
  export let onAttachURL = async () => false;
  export let onKeydown = () => {};
  export let onPaste = () => {};
  export let onStop = () => {};
  export let onSend = () => {};

  let sourceOpen = false;
  let sourceURL = '';

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
</script>

<footer>
  {#if pendingAttachments.length || uploadingAttachments}
    <div class="pending-attachments">
      {#each pendingAttachments as attachment}
        <div class="pending-attachment">
          {#if attachmentKind(attachment) === 'image'}
            <img src={attachment.url} alt={attachment.name} />
          {:else if attachmentKind(attachment) === 'video' && canPreviewVideo(attachment)}
            <video src={attachment.url} muted preload="metadata" aria-label={attachment.name}></video>
          {:else}
            <span class="media-file-icon">{attachmentKind(attachment) === 'audio' ? '♪' : '▶'}</span>
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
  <div class="composer" role="group" aria-label="메시지와 미디어 입력">
    <input class="media-input" bind:this={attachmentInput} type="file" accept={attachmentAccept} multiple onchange={onAttachmentInputChange} />
    <button class="attach" onclick={() => attachmentInput?.click()} disabled={!activeId || running || uploadingAttachments || pendingAttachments.length >= 6} aria-label="미디어 첨부" title="이미지·음성·비디오 첨부">＋</button>
    <button class="attach attach-url" onclick={() => sourceOpen = !sourceOpen} disabled={!activeId || running || uploadingAttachments || pendingAttachments.length >= 6} aria-label="URL 미디어 첨부" title="YouTube 등 URL에서 미디어 가져오기">⌁</button>
    <textarea bind:value={input} onkeydown={onKeydown} onpaste={onPaste} placeholder={activeId ? '메시지를 입력하세요' : '새 대화를 만든 뒤 메시지를 입력하세요'} rows="1" disabled={!activeId || running}></textarea>
    {#if running}<button class="send stop" onclick={onStop}>■</button>{:else}<button class="send" onclick={onSend} disabled={!activeId || !input.trim() || uploadingAttachments}>↑</button>{/if}
  </div>
  <small>파일 드래그 또는 URL 영상 첨부 · Enter 전송 · Shift+Enter 줄바꿈 · reasoning: {reasoningEffort || '서버 기본값'} · 웹: {webToolsEnabled ? '자동' : '꺼짐'}</small>
</footer>
