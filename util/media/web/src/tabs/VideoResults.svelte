<script>
  import ResultPagination from '../ResultPagination.svelte'
  import { imagePageSizeOptions } from '../lib/catalogs.js'
  import { videoAccelerationLabel, videoFPSLabel, videoJobDuration } from '../lib/mediaPresentation.js'
  import { formatDuration } from '../lib/videoTiming.js'

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
  export let progressFor = () => null
  export let promptText = () => ''
  export let onView = () => {}
  export let onPage = () => {}
  export let onPageSize = () => {}
  export let onSort = () => {}
  export let onShow = () => {}
  export let onPrompt = () => {}
  export let onShowUpscaleSource = () => {}
  export let onLoadSettings = () => {}
  export let onFrame = () => {}
  export let onUpscale = () => {}
  export let onSendToRecognition = () => {}
  export let onCancel = () => {}
  export let onRetry = () => {}
  export let onDelete = () => {}

  function detail(job) {
    return `영상 · ${job.params?.width}×${job.params?.height} · ${formatDuration(videoJobDuration(job))}${videoFPSLabel(job)}`
  }
</script>

<aside class="video-results-pane mobile-results-pane">
  <div class="results-heading">
    <h3>생성 영상 목록</h3>
    <div class="view-switch" aria-label="생성 영상 목록 보기 방식">
      <button type="button" class:active={view === 'gallery'} onclick={() => onView('gallery')}>갤러리</button>
      <button type="button" class:active={view === 'list'} onclick={() => onView('list')}>리스트</button>
    </div>
  </div>
  <ResultPagination label="생성 영상 목록" total={jobs.length} {page} {pageSize} pageSizes={imagePageSizeOptions} {sortOrder} onPageChange={onPage} onPageSizeChange={onPageSize} onSortOrderChange={onSort} />
  <div class="video-list" class:list-view={view === 'list'}>
    {#each pagedJobs as job, videoIndex (job.id)}
      {@const generationProgress = job.status === 'queued' || job.status === 'running' ? progressFor(job) : null}
      {@const visibleIndex = (page - 1) * pageSize + videoIndex + 1}
      <article class:pending={job.status !== 'completed'}>
        <span class="result-index-badge" title={`대화창에서 ${visibleIndex}번 영상으로 지칭`}>#{visibleIndex}</span>
        {#if view === 'list'}
          {#if job.output_url}
            <button type="button" class="video-list-thumb" aria-label="영상 크게 보기" onclick={() => onShow(job)}><!-- svelte-ignore a11y_media_has_caption --><video preload="metadata" muted playsinline src={job.output_url}></video></button>
          {:else}
            <div class="video-list-thumb empty-thumb">{#if generationProgress}<div class="image-generation-status compact"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small></div>{:else}<span>{job.status}</span>{/if}</div>
          {/if}
          <div class="video-list-content"><small>{job.params?.width}×{job.params?.height} · {formatDuration(videoJobDuration(job))}{videoFPSLabel(job)}{#if videoAccelerationLabel(job)} · {videoAccelerationLabel(job)}{/if}{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</small><button type="button" class="image-prompt" title={job.prompt} onclick={() => onPrompt(job, detail(job), promptText(job))}>{job.prompt}</button>{#if job.error}<em>{job.error}</em>{/if}</div>
        {:else}
          {#if job.output_url}
            <button type="button" class="video-gallery-thumb" aria-label="영상 크게 보기" title="클릭하여 크게 보기" onclick={() => onShow(job)}><!-- svelte-ignore a11y_media_has_caption --><video preload="metadata" muted playsinline src={job.output_url}></video></button>
          {:else}
            <div class="video-placeholder">{#if generationProgress}<div class="image-generation-status"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small>{#if generationProgress.eta}<small>{generationProgress.eta}</small>{/if}</div>{:else}<span>{job.status}</span>{/if}</div>
          {/if}
          <button type="button" class="image-prompt" title={job.prompt} onclick={() => onPrompt(job, detail(job), promptText(job))}>{job.prompt}</button><small>{job.params?.width}×{job.params?.height} · {formatDuration(videoJobDuration(job))}{videoFPSLabel(job)}{#if videoAccelerationLabel(job)} · {videoAccelerationLabel(job)}{/if}{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</small>{#if job.error}<em>{job.error}</em>{/if}
        {/if}
        {#if job.status === 'completed' && job.output_url}
          <div class="video-utility-actions"><span>활용:</span>{#if job.params?.mode === 'upscale'}<button type="button" onclick={() => onShowUpscaleSource(job)}>원본</button>{:else}<button type="button" onclick={() => onLoadSettings(job)}>설정</button>{/if}<button type="button" onclick={() => onFrame(job)}>장면</button><button type="button" disabled={!upscaleOnline} onclick={() => onUpscale(job)}>업스케일</button><button type="button" onclick={() => onSendToRecognition(job)}>자막 생성</button></div>
        {/if}
        {#if job.status === 'queued' || job.status === 'running'}
          <div class="video-job-actions"><button class="job-stop" disabled={cancellingJob === job.id} onclick={() => onCancel(job)}>{cancellingJob === job.id ? (job.status === 'running' ? '중지 중…' : '취소 중…') : (job.status === 'running' ? '중지' : '대기 취소')}</button></div>
        {:else if job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'}
          <div class="video-job-actions">{#if job.status === 'failed' || job.status === 'cancelled'}<button type="button" class="job-retry" disabled={retryingJob === job.id} onclick={() => onRetry(job)}>{retryingJob === job.id ? '재시도 중…' : '재시도'}</button>{/if}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => onDelete(job)}>삭제</button></div>
        {/if}
      </article>
    {:else}
      <div class="empty">첫 영상이 여기에 나타납니다.</div>
    {/each}
  </div>
  <ResultPagination compact label="생성 영상 목록" total={jobs.length} {page} {pageSize} onPageChange={onPage} />
</aside>
