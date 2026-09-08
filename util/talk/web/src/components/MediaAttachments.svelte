<script>
  import { attachmentKind, canPreviewVideo, formatAttachmentSize } from '../lib/attachments.js';

  export let attachments = [];
  let previewDialog;
  let preview = null;
  function showPreview(item) { preview = item; previewDialog.showModal(); }
</script>

<div class="media-gallery">
  {#each attachments as attachment}
    {@const kind = attachmentKind(attachment)}
    {#if kind === 'image'}
      <a class="media-image" href={attachment.url} target="_blank" rel="noreferrer" title={attachment.name}>
        <img src={attachment.url} alt={attachment.name} loading="lazy" />
      </a>
    {:else if kind === 'audio'}
      <div class="media-player audio-player">
        <span><strong>♪ {attachment.name}</strong><small>{formatAttachmentSize(attachment.size)}</small></span>
        <audio src={attachment.url} controls preload="metadata"></audio>
      </div>
	{:else if kind === 'document'}
        <div class="document-card">
          <a class="media-file document-file" href={attachment.url} download={attachment.name} title={attachment.name}>
            <span>▤</span><strong>{attachment.name}</strong><small>{formatAttachmentSize(attachment.size)} · 다운로드</small>
          </a>
          {#if attachment.mime === 'application/pdf'}<button type="button" onclick={() => showPreview(attachment)}>PDF 보기</button>{/if}
        </div>
    {:else if canPreviewVideo(attachment)}
      <div class="media-player video-player">
        <!-- svelte-ignore a11y_media_has_caption: user-provided video has no separate caption track -->
        <video src={attachment.url} controls preload="metadata" aria-label={attachment.name}></video>
        <span><strong>{attachment.name}</strong><small>{formatAttachmentSize(attachment.size)}</small></span>
      </div>
    {:else}
      <a class="media-file" href={attachment.url} target="_blank" rel="noreferrer" title={attachment.name}>
        <span>▶</span><strong>{attachment.name}</strong><small>{formatAttachmentSize(attachment.size)} · 브라우저 미리보기 미지원</small>
      </a>
    {/if}
  {/each}
</div>

<dialog bind:this={previewDialog} class="document-preview" onclose={() => { preview = null; }}>
  <header><strong>{preview?.name || '문서 미리보기'}</strong><button type="button" aria-label="문서 미리보기 닫기" onclick={() => previewDialog.close()}>×</button></header>
  {#if preview}<iframe title="PDF 보기" src={preview.url}></iframe><p>Office와 함께 생성된 PDF는 원본과 배치가 다를 수 있습니다. 표시되지 않으면 파일을 다운로드하세요.</p>{/if}
</dialog>
<style>
.document-card{display:flex;align-items:center;gap:8px;max-width:100%}.document-card a{min-width:0}.document-card button{white-space:nowrap;border:1px solid #52617a;border-radius:6px;padding:6px 10px;background:transparent;color:inherit}
.document-preview{width:min(1000px,calc(100vw - 32px));height:calc(100dvh - 48px);box-sizing:border-box;border:1px solid #52617a;border-radius:12px;background:#161d29;color:#edf1f7;padding:16px}.document-preview::backdrop{background:#0009}.document-preview header{display:flex;justify-content:space-between;align-items:center;gap:12px}.document-preview header strong{overflow-wrap:anywhere}.document-preview button{background:transparent;border:0;color:inherit;font-size:24px}.document-preview iframe{width:100%;height:calc(100% - 90px);border:0;margin-top:12px;background:white}.document-preview p{font-size:12px;opacity:.75}
:global(html[data-theme="light"]) .document-preview{background:white;color:#27354a}
</style>
