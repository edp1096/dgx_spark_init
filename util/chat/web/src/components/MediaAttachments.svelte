<script>
  import { attachmentKind, canPreviewVideo, formatAttachmentSize } from '../lib/attachments.js';

  export let attachments = [];
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
		<a class="media-file document-file" href={attachment.url} target="_blank" rel="noreferrer" title={attachment.name}>
			<span>▤</span><strong>{attachment.name}</strong><small>{formatAttachmentSize(attachment.size)} · 문서</small>
		</a>
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
