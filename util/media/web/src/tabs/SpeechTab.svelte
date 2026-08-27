<script>
  import ResultPagination from '../ResultPagination.svelte'
  import { imagePageSizeOptions, statusLabels } from '../lib/catalogs.js'

  export let mobilePane = 'create'
  export let form
  export let busy = false
  export let activeJobs = []
  export let jobs = []
  export let pagedJobs = []
  export let view = 'gallery'
  export let page = 1
  export let pageSize = 10
  export let sortOrder = 'desc'
  export let cancellingJob = ''
  export let retryingJob = ''
  export let deletingJob = ''
  export let progressFor = () => null
  export let onMobilePane = () => {}
  export let onReset = () => {}
  export let onGenerate = () => {}
  export let onView = () => {}
  export let onPage = () => {}
  export let onPageSize = () => {}
  export let onSort = () => {}
  export let onPrompt = () => {}
  export let onShow = () => {}
  export let onSendToVideo = () => {}
  export let onCancel = () => {}
  export let onRetry = () => {}
  export let onDelete = () => {}
</script>

<div class="mobile-image-nav" role="tablist" aria-label="모바일 음성 화면">
  <button type="button" role="tab" aria-selected={mobilePane === 'create'} class:active={mobilePane === 'create'} onclick={() => onMobilePane('create')}><span>만들기</span><small>음성 생성 설정</small></button>
  <button type="button" role="tab" aria-selected={mobilePane === 'results'} class:active={mobilePane === 'results'} onclick={() => onMobilePane('results')}><span>생성 음성 목록</span><small>{jobs.length}개{#if activeJobs.some((job) => job.kind === 'speech')} · 생성 중{/if}</small></button>
</div>
<section class="workspace mobile-media-workspace" class:mobile-results={mobilePane === 'results'}>
  <form class="mobile-create-pane" onsubmit={(event) => { event.preventDefault(); onGenerate() }}>
    <div class="section-title"><div><span>03</span><h2>음성 생성</h2></div><div class="image-title-actions"><button type="button" class="quiet image-create-reset" disabled={busy} title="음성 생성 설정을 모두 비웁니다." onclick={onReset}>초기화</button></div></div>
    <label>읽을 문장<textarea bind:value={form.text} rows="7" placeholder="음성으로 변환할 문장을 입력하세요." required></textarea></label>
    <label>연기 지시 <small>선택 사항 · 1.7B instruction control</small><textarea bind:value={form.instructions} rows="3" placeholder="예: 기쁘고 활기찬 목소리로, 중요한 단어는 힘주어 말해 주세요."></textarea></label>
    <div class="fields three">
      <label>언어<select bind:value={form.language}><option>Korean</option><option>English</option><option>Chinese</option><option>Japanese</option><option>Auto</option></select></label>
      <label>화자<select bind:value={form.speaker}><option>Sohee</option><option>Vivian</option><option>Serena</option><option>Ryan</option><option>Aiden</option><option>Ono_Anna</option></select></label>
      <label><span>시드 <small>-1은 무작위</small></span><input type="number" min="-1" bind:value={form.seed}></label>
    </div>
    <button class="primary" disabled={busy}>{busy ? '요청 중…' : activeJobs.some((job) => job.kind === 'image' || job.kind === 'video' || job.kind === 'speech') ? '음성 큐에 추가' : '음성 만들기'}</button>
  </form>
  <aside class="speech-results-pane mobile-results-pane">
    <div class="results-heading"><h3>생성 음성 목록</h3><div class="view-switch" aria-label="생성 음성 목록 보기 방식"><button type="button" class:active={view === 'gallery'} onclick={() => onView('gallery')}>갤러리</button><button type="button" class:active={view === 'list'} onclick={() => onView('list')}>리스트</button></div></div>
    <ResultPagination label="생성 음성 목록" total={jobs.length} {page} {pageSize} pageSizes={imagePageSizeOptions} {sortOrder} onPageChange={onPage} onPageSizeChange={onPageSize} onSortOrderChange={onSort} />
    <div class="speech-list" class:list-view={view === 'list'}>
      {#each pagedJobs as job, speechIndex (job.id)}
        {@const visibleSpeechIndex = (page - 1) * pageSize + speechIndex + 1}
        {@const generationProgress = job.status === 'queued' || job.status === 'running' ? progressFor(job) : null}
        <article class:pending={job.status !== 'completed'}>
          <span class="result-index-badge" title={`대화창에서 ${visibleSpeechIndex}번 음성으로 지칭`}>#{visibleSpeechIndex}</span>
          <div class="speech-card-heading"><span>{job.params?.speaker || 'VOICE'}{#if job.params?.language} · {job.params.language}{/if}</span>{#if job.output_url}<button type="button" class="audio-modal-open" onclick={() => onShow(job)}>크게 보기</button>{/if}</div>
          <button type="button" class="speech-prompt" title={job.prompt} onclick={() => onPrompt(job)}>{job.prompt || '원문 없음'}</button>
          <small class="speech-meta">{#if job.params?.seed >= 0}seed {job.params.seed}{:else}무작위 시드{/if}{#if job.params?.instructions} · 지시 있음{/if}</small>
          {#if job.params?.instructions}<small class="instruction" title={job.params.instructions}>지시 · {job.params.instructions}</small>{/if}
          {#if job.output_url}
            <audio controls src={job.output_url}></audio><div class="audio-utility-actions"><span>활용:</span><button type="button" onclick={() => onSendToVideo(job)}>영상 생성</button></div>
          {:else if generationProgress}
            <div class="image-generation-status speech-generation-status"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small>{#if generationProgress.eta}<small>{generationProgress.eta}</small>{/if}</div>
          {:else}
            <small class="speech-status">{statusLabels[job.status] || job.status}</small>
          {/if}
          {#if job.error}<em>{job.error}</em>{/if}
          <div class="speech-job-actions">
            {#if job.status === 'queued' || job.status === 'running'}
              <button class="job-stop" disabled={cancellingJob === job.id} onclick={() => onCancel(job)}>{cancellingJob === job.id ? (job.status === 'running' ? '중지 중…' : '취소 중…') : (job.status === 'running' ? '중지' : '대기 취소')}</button>
            {:else if job.status === 'failed' || job.status === 'cancelled'}
              <button class="job-retry" disabled={retryingJob === job.id} onclick={() => onRetry(job)}>{retryingJob === job.id ? '재시도 중…' : '재시도'}</button><button class="job-delete" disabled={deletingJob === job.id} onclick={() => onDelete(job)}>삭제</button>
            {:else if job.status === 'completed'}
              <button class="job-delete" disabled={deletingJob === job.id} onclick={() => onDelete(job)}>삭제</button>
            {/if}
          </div>
        </article>
      {:else}
        <div class="empty">첫 음성이 여기에 나타납니다.</div>
      {/each}
    </div>
    <ResultPagination compact label="생성 음성 목록" total={jobs.length} {page} {pageSize} onPageChange={onPage} />
  </aside>
</section>
