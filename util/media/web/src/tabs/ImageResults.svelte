<script>
  import ResultPagination from '../ResultPagination.svelte'
  import { imageModeMeta, imagePageSizeOptions } from '../lib/catalogs.js'
  import { imageModuleSummary, imageSamplingSummary } from '../lib/mediaPresentation.js'

  export let jobs = []
  export let pagedJobs = []
  export let view = 'gallery'
  export let page = 1
  export let pageSize = 10
  export let sortOrder = 'desc'
  export let garmentOnline = false
  export let imageOnline = false
  export let upscaleOnline = false
  export let cloningJob = ''
  export let detailEnhancingJob = ''
  export let upscalingJob = ''
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
  export let onClone = () => {}
  export let onContinueEditing = () => {}
  export let onGarment = () => {}
  export let onDetail = () => {}
  export let onUpscale = () => {}
  export let onCancel = () => {}
  export let onRetry = () => {}
  export let onDelete = () => {}

  function modeLabel(job) {
    return imageModeMeta[job.params?.mode]?.label || '이미지'
  }

  function promptDetail(job) {
    return `${modeLabel(job)} · ${job.params?.width || '—'}×${job.params?.height || '—'}${imageSamplingSummary(job) ? ` · ${imageSamplingSummary(job)}` : ''}`
  }
</script>

<aside class="image-results-pane">
  <div class="results-heading">
    <h3>생성 이미지 목록</h3>
    <div class="view-switch" aria-label="생성 이미지 목록 보기 방식">
      <button type="button" class:active={view === 'gallery'} onclick={() => onView('gallery')}>갤러리</button>
      <button type="button" class:active={view === 'list'} onclick={() => onView('list')}>리스트</button>
    </div>
  </div>
  <ResultPagination label="생성 이미지 목록" total={jobs.length} {page} {pageSize} pageSizes={imagePageSizeOptions} {sortOrder} onPageChange={onPage} onPageSizeChange={onPageSize} onSortOrderChange={onSort} />
  <div class="gallery image-results" class:list-view={view === 'list'}>
    {#each pagedJobs as job, imageIndex (job.id)}
      {@const generationProgress = job.status === 'queued' || job.status === 'running' ? progressFor(job) : null}
      {@const visibleIndex = (page - 1) * pageSize + imageIndex + 1}
      <article class:pending={job.status !== 'completed'}>
        <span class="image-index-badge" title={`대화창에서 ${visibleIndex}번 이미지로 지칭`}>#{visibleIndex}</span>
        {#if view === 'list'}
          {#if job.output_url}
            <button type="button" class="image-list-thumb image-zoom" aria-label="생성 이미지 크게 보기" onclick={(event) => onShow(event, job.output_url, '생성 이미지', job.prompt, job.id)}><img src={job.output_url} alt={job.prompt}></button>
          {:else}
            <div class="image-list-thumb placeholder">{#if generationProgress}<div class="image-generation-status compact"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small></div>{:else}<span>{job.status}</span>{/if}</div>
          {/if}
          <div class="image-list-content">
            <span>{modeLabel(job)}{imageModuleSummary(job)} · {job.params?.width || '—'}×{job.params?.height || '—'}{#if imageSamplingSummary(job)} · {imageSamplingSummary(job)}{/if}{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</span>
            <button type="button" class="image-prompt" title="클릭하여 전체 프롬프트 보기" onclick={() => onPrompt(job, promptDetail(job), promptText(job))}>{job.prompt}</button>
            {#if job.error}<em>{job.error}</em>{/if}
            {#if job.status === 'failed' || job.status === 'cancelled'}<button type="button" class="job-retry image-retry" disabled={retryingJob === job.id} onclick={() => onRetry(job)}>{retryingJob === job.id ? '재시도 중…' : '재시도'}</button>{/if}
            <div class="image-clone-actions" aria-label="이 작업에서 불러오기">
              <span>불러오기:</span>
              <button type="button" disabled={Boolean(cloningJob)} onclick={() => onClone(job, 'prompt')}>프롬프트</button>
              <button type="button" disabled={Boolean(cloningJob)} onclick={() => onClone(job, 'references')}>참조</button>
              <button type="button" disabled={Boolean(cloningJob)} onclick={() => onClone(job, 'settings')}>설정</button>
              <button type="button" class="clone-all" disabled={Boolean(cloningJob)} onclick={() => onClone(job, 'all')}>{cloningJob === `${job.id}:all` ? '불러오는 중…' : '전체'}</button>
            </div>
            {#if job.status === 'completed'}
              <div class="image-post-actions"><span>후처리:</span><button type="button" title="이 결과를 Identity 원본으로 불러와 계속 편집" onclick={() => onContinueEditing(job)}>편집</button>{#if job.params?.mode === 'garment_extract' && job.outputs?.mask}<button type="button" title="저장된 의상 마스크 보기" onclick={(event) => onShow(event, job.outputs.mask, '의상 마스크', job.prompt, job.id)}>마스크</button>{:else}<button type="button" title="이 이미지에서 의상만 투명 PNG로 추출" disabled={!garmentOnline} onclick={() => onGarment(job)}>의상</button>{/if}<button type="button" title="Ostris Edit LoRA로 다시 그립니다. 얼굴·색·글자·구도가 달라질 수 있습니다." disabled={Boolean(detailEnhancingJob) || Boolean(upscalingJob) || !imageOnline} onclick={() => onDetail(job)}>{detailEnhancingJob === job.id ? '처리 중…' : '디테일'}</button><button type="button" title="SeedVR2로 복원하고 2배 확대" disabled={Boolean(detailEnhancingJob) || Boolean(upscalingJob) || !upscaleOnline} onclick={() => onUpscale(job)}>{upscalingJob === job.id ? '처리 중…' : '업스케일'}</button></div>
            {/if}
          </div>
        {:else}
          {#if job.output_url}
            <button type="button" class="gallery-image image-zoom" aria-label="생성 이미지 크게 보기" onclick={(event) => onShow(event, job.output_url, '생성 이미지', job.prompt, job.id)}><img src={job.output_url} alt={job.prompt}></button>
          {:else}
            <div class="placeholder">{#if generationProgress}<div class="image-generation-status"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small>{#if generationProgress.eta}<small>{generationProgress.eta}</small>{/if}</div>{:else}<span>{job.status}</span>{/if}</div>
          {/if}
          <span class="image-mode-badge" title={`${modeLabel(job)}${imageModuleSummary(job)}`}>{modeLabel(job)}{imageModuleSummary(job)}</span>
          <button type="button" class="image-prompt" title="클릭하여 전체 프롬프트 보기" onclick={() => onPrompt(job, promptDetail(job), promptText(job))}>{job.prompt}</button>
          {#if job.error}<em>{job.error}</em>{/if}
          {#if job.status === 'failed' || job.status === 'cancelled'}<button type="button" class="job-retry image-retry" disabled={retryingJob === job.id} onclick={() => onRetry(job)}>{retryingJob === job.id ? '재시도 중…' : '재시도'}</button>{/if}
          <div class="image-clone-actions" aria-label="이 작업에서 불러오기">
            <span>불러오기:</span>
            <button type="button" disabled={Boolean(cloningJob)} onclick={() => onClone(job, 'prompt')}>프롬프트</button>
            <button type="button" disabled={Boolean(cloningJob)} onclick={() => onClone(job, 'references')}>참조</button>
            <button type="button" disabled={Boolean(cloningJob)} onclick={() => onClone(job, 'settings')}>설정</button>
            <button type="button" class="clone-all" disabled={Boolean(cloningJob)} onclick={() => onClone(job, 'all')}>{cloningJob === `${job.id}:all` ? '불러오는 중…' : '전체'}</button>
          </div>
          {#if job.status === 'completed'}
            <div class="image-post-actions"><span>후처리:</span><button type="button" title="이 결과를 Identity 원본으로 불러와 계속 편집" onclick={() => onContinueEditing(job)}>편집</button>{#if job.params?.mode === 'garment_extract' && job.outputs?.mask}<button type="button" title="저장된 의상 마스크 보기" onclick={(event) => onShow(event, job.outputs.mask, '의상 마스크', job.prompt, job.id)}>마스크</button>{:else}<button type="button" title="이 이미지에서 의상만 투명 PNG로 추출" disabled={!garmentOnline} onclick={() => onGarment(job)}>의상</button>{/if}<button type="button" title="Ostris Edit LoRA로 다시 그립니다. 얼굴·색·글자·구도가 달라질 수 있습니다." disabled={Boolean(detailEnhancingJob) || Boolean(upscalingJob) || !imageOnline} onclick={() => onDetail(job)}>{detailEnhancingJob === job.id ? '처리 중…' : '디테일'}</button><button type="button" title="SeedVR2로 복원하고 2배 확대" disabled={Boolean(detailEnhancingJob) || Boolean(upscalingJob) || !upscaleOnline} onclick={() => onUpscale(job)}>{upscalingJob === job.id ? '처리 중…' : '업스케일'}</button></div>
          {/if}
        {/if}
        {#if job.status === 'queued' || job.status === 'running'}
          <button class="job-stop" disabled={cancellingJob === job.id} onclick={() => onCancel(job)}>{cancellingJob === job.id ? (job.status === 'running' ? '중지 중…' : '취소 중…') : (job.status === 'running' ? '중지' : '대기 취소')}</button>
        {:else if job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'}
          <button class="job-delete" disabled={deletingJob === job.id} onclick={() => onDelete(job)}>삭제</button>
        {/if}
      </article>
    {:else}
      <div class="empty">첫 이미지가 여기에 나타납니다.</div>
    {/each}
  </div>
  <ResultPagination compact label="생성 이미지 목록" total={jobs.length} {page} {pageSize} onPageChange={onPage} />
</aside>
