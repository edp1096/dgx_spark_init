<script>
  import ResultPagination from '../ResultPagination.svelte'
  import { imagePageSizeOptions, outputLabels, statusLabels } from '../lib/catalogs.js'
  import {
    isAudioMedia,
    mediaSummary,
    recognitionLanguageLabel,
    subtitleTranslationWarnings,
  } from '../lib/mediaPresentation.js'

  export let jobs = []
  export let pagedJobs = []
  export let view = 'gallery'
  export let page = 1
  export let pageSize = 10
  export let sortOrder = 'desc'
  export let upscaleOnline = false
  export let cancellingJob = ''
  export let retryingJob = ''
  export let deletingJob = ''
  export let progressText = () => ''
  export let progressTiming = () => ''
  export let progressPercent = () => 0
  export let warningText = () => ''
  export let onView = () => {}
  export let onPage = () => {}
  export let onPageSize = () => {}
  export let onSort = () => {}
  export let onShow = () => {}
  export let onWarning = () => {}
  export let onRegenerate = () => {}
  export let onFrame = () => {}
  export let onUpscale = () => {}
  export let onCancel = () => {}
  export let onRetry = () => {}
  export let onDelete = () => {}
</script>

<aside class="subtitle-results-pane mobile-results-pane">
  <div class="results-heading">
    <h3>생성 자막 목록</h3>
    <div class="view-switch" aria-label="생성 자막 목록 보기 방식">
      <button type="button" class:active={view === 'gallery'} onclick={() => onView('gallery')}>갤러리</button>
      <button type="button" class:active={view === 'list'} onclick={() => onView('list')}>리스트</button>
    </div>
  </div>
  <ResultPagination
    label="생성 자막 목록"
    total={jobs.length}
    {page}
    {pageSize}
    pageSizes={imagePageSizeOptions}
    {sortOrder}
    onPageChange={onPage}
    onPageSizeChange={onPageSize}
    onSortOrderChange={onSort}
  />
  <div class="audio-list subtitle-results" class:list-view={view === 'list'}>
    {#each pagedJobs as job, recognitionIndex (job.id)}
      {@const visibleIndex = (page - 1) * pageSize + recognitionIndex + 1}
      {@const warnings = subtitleTranslationWarnings(job)}
      <article class:pending={job.status === 'queued' || job.status === 'running'}>
        <span class="result-index-badge" title={`대화창에서 ${visibleIndex}번 받아쓰기 결과로 지칭`}>#{visibleIndex}</span>
        {#if view === 'list'}
          {#if job.media_url || job.params?.text}
            <button type="button" class="subtitle-list-thumb" class:empty-thumb={!job.media_url || isAudioMedia(job)} aria-label="자막 결과 크게 보기" onclick={() => onShow(job)}>
              {#if job.media_url && !isAudioMedia(job)}
                <!-- svelte-ignore a11y_media_has_caption -->
                <video preload="metadata" muted playsinline src={job.media_url}></video>
              {:else}
                <span>{job.media_url && isAudioMedia(job) ? 'AUDIO' : job.params?.text ? 'TEXT' : job.status}</span>
              {/if}
            </button>
          {:else}
            <div class="subtitle-list-thumb empty-thumb"><span>{job.status}</span></div>
          {/if}
        {:else if job.media_url || job.params?.text}
          <button type="button" class="subtitle-gallery-thumb" class:empty-thumb={!job.media_url || isAudioMedia(job)} aria-label="자막 결과 크게 보기" onclick={() => onShow(job)}>
            {#if job.media_url && !isAudioMedia(job)}
              <!-- svelte-ignore a11y_media_has_caption -->
              <video preload="metadata" muted playsinline src={job.media_url}></video>
            {:else}
              <span>{job.media_url && isAudioMedia(job) ? 'AUDIO' : job.params?.text ? 'TEXT' : statusLabels[job.status] || job.status}</span>
            {/if}
          </button>
        {:else}
          <div class="subtitle-gallery-thumb empty-thumb"><span>{statusLabels[job.status] || job.status}</span></div>
        {/if}

        <div class="subtitle-result-title">
          <span>{job.params?.detected_language || recognitionLanguageLabel(job.params?.language)}{#if job.params?.segments} · {job.params.segments}구간{/if}{#if job.params?.media_part} · 파트 {job.params.media_part}{/if}{#if job.params?.media_source} · {job.params.media_source}{/if}</span>
          <p title={job.prompt}>{job.prompt}</p>
          {#if job.params?.media}<small>{mediaSummary(job)}</small>{/if}
        </div>
        {#if warnings.length}
          <button type="button" class="subtitle-translation-warning" onclick={() => onWarning(job, warnings, warningText(job))}>번역 경고 {warnings.length}개</button>
        {/if}
        {#if !job.params?.text}
          <small class="recognition-progress-text">{progressText(job)}{#if progressTiming(job)} · {progressTiming(job)}{/if}</small>
        {/if}
        {#if job.status === 'queued' || job.status === 'running'}
          <div class="recognition-progress" aria-label={progressText(job)}><i style={`width: ${progressPercent(job)}%`}></i></div>
        {/if}
        {#if job.outputs}
          <div class="output-links">
            {#each Object.entries(job.outputs) as output}<a href={output[1]} target="_blank">{outputLabels[output[0]] || output[0]} ↗</a>{/each}
            {#if job.status === 'completed'}<button type="button" class="subtitle-regenerate-open" onclick={() => onRegenerate(job)}>자막 재생성</button>{/if}
          </div>
        {:else if job.output_url}
          <a href={job.output_url} target="_blank">결과 열기 ↗</a>
        {/if}
        {#if job.media_url && !isAudioMedia(job)}
          <div class="subtitle-video-actions"><span>영상 활용:</span><button type="button" onclick={() => onFrame(job)}>장면 선택</button><button type="button" disabled={!upscaleOnline} onclick={() => onUpscale(job)}>업스케일</button></div>
        {/if}
        {#if job.error}<em>{job.error}</em>{/if}
        {#if job.status === 'queued' || job.status === 'running'}
          <button class="job-stop" disabled={cancellingJob === job.id} onclick={() => onCancel(job)}>{cancellingJob === job.id ? '중지 중…' : job.status === 'queued' ? '대기 취소' : '중지'}</button>
        {:else}
          <div class="job-actions">
            {#if job.status === 'failed' || job.status === 'cancelled'}<button class="job-retry" disabled={retryingJob === job.id} onclick={() => onRetry(job)}>{retryingJob === job.id ? '재개 중…' : '재개'}</button>{/if}
            <button class="job-delete" disabled={deletingJob === job.id} onclick={() => onDelete(job)}>삭제</button>
          </div>
        {/if}
      </article>
    {:else}
      <div class="empty">첫 자막 작업이 여기에 나타납니다.</div>
    {/each}
  </div>
  <ResultPagination compact label="생성 자막 목록" total={jobs.length} {page} {pageSize} onPageChange={onPage} />
</aside>
