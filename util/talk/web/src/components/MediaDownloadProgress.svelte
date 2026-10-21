<script>
  import { downloadPercent, downloadStreamLabel, mediaDownloadLabel } from '../lib/media-download-progress.js';
  export let tool;
</script>

<section class="media-download-progress" role="status" aria-live="polite" aria-label="미디어 다운로드 진행 상태">
  <small>{tool.progress?.stage === 'downloading' ? '영상·음성 다운로드 중' : mediaDownloadLabel(tool)}</small>
  {#each tool.progress?.streams || [] as stream (stream.id)}
    <div class="download-stream">
      <small>{downloadStreamLabel(stream)}</small>
      {#if stream.status !== 'queued'}
        {#if downloadPercent(stream) !== null}
          <progress max="100" value={downloadPercent(stream)} aria-label={stream.kind === 'audio' ? '음성 다운로드 진행률' : '영상 다운로드 진행률'}></progress>
        {:else}
          <progress aria-label="다운로드 진행 중"></progress>
        {/if}
      {/if}
    </div>
  {/each}
</section>

<style>
  .media-download-progress { display: grid; gap: .55rem; margin: .55rem 0; padding: .65rem .8rem; border: 1px solid var(--border, #8884); border-radius: .6rem; }
  .download-stream { display: grid; gap: .2rem; }
  small { overflow-wrap: anywhere; }
  progress { width: 100%; height: .45rem; accent-color: var(--accent, #60a5fa); }
</style>
