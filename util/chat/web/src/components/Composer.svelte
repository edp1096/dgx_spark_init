<script>
  import { attachmentAccept, attachmentKind, canPreviewVideo, formatAttachmentSize } from '../lib/attachments.js';

  export let pendingAttachments = [];
  export let uploadingAttachments = false;
  export let running = false;
  export let activeId = '';
  export let input = '';
  export let attachmentInput;
  export let reasoningEffort = '';
  export let webToolsEnabled = false;
  export let onRemoveAttachment = () => {};
  export let onAttachmentInputChange = () => {};
  export let onKeydown = () => {};
  export let onPaste = () => {};
  export let onStop = () => {};
  export let onSend = () => {};
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
      {#if uploadingAttachments}<span class="uploading">미디어 업로드 중…</span>{/if}
    </div>
  {/if}
  <div class="composer" role="group" aria-label="메시지와 미디어 입력">
    <input class="media-input" bind:this={attachmentInput} type="file" accept={attachmentAccept} multiple onchange={onAttachmentInputChange} />
    <button class="attach" onclick={() => attachmentInput?.click()} disabled={!activeId || running || uploadingAttachments || pendingAttachments.length >= 6} aria-label="미디어 첨부" title="이미지·음성·비디오 첨부">＋</button>
    <textarea bind:value={input} onkeydown={onKeydown} onpaste={onPaste} placeholder={activeId ? '메시지를 입력하세요' : '새 대화를 만든 뒤 메시지를 입력하세요'} rows="1" disabled={!activeId || running}></textarea>
    {#if running}<button class="send stop" onclick={onStop}>■</button>{:else}<button class="send" onclick={onSend} disabled={!activeId || !input.trim() || uploadingAttachments}>↑</button>{/if}
  </div>
  <small>이미지·음성·비디오 드래그 가능 · Enter 전송 · Shift+Enter 줄바꿈 · reasoning: {reasoningEffort || '서버 기본값'} · 웹: {webToolsEnabled ? '자동' : '꺼짐'}</small>
</footer>
