<script>
  import ResultPagination from '../ResultPagination.svelte'
  import { kindLabels, pageSizeOptions, statusLabels } from '../lib/catalogs.js'

  export let jobs = []
  export let pagedJobs = []
  export let page = 1
  export let pageSize = 20
  export let sortOrder = 'desc'
  export let deletingJob = ''
  export let retryingJob = ''
  export let activeJobs = []
  export let onClear = () => {}
  export let onPage = () => {}
  export let onPageSize = () => {}
  export let onSort = () => {}
  export let onPrompt = () => {}
  export let onRetry = () => {}
  export let onDelete = () => {}
</script>

<section class="history">
  <div class="section-title">
    <div><span>06</span><h2>생성 기록</h2></div>
    {#if jobs.some((job) => job.status !== 'queued' && job.status !== 'running')}
      <button class="quiet danger" disabled={deletingJob === 'all'} onclick={onClear}>모두 비우기</button>
    {/if}
  </div>
  <ResultPagination label="생성 기록" total={jobs.length} {page} {pageSize} pageSizes={pageSizeOptions} {sortOrder} onPageChange={onPage} onPageSizeChange={onPageSize} onSortOrderChange={onSort} />
  {#each pagedJobs as job (job.id)}
    <article>
      <span class="kind">{kindLabels[job.kind] || job.kind}</span>
      <div>
        <button type="button" class="history-prompt" title="전체 내용 보기" onclick={() => onPrompt(job)}>{job.prompt}</button>
        <small>{new Date(job.created_at).toLocaleString()} · {statusLabels[job.status] || job.status}</small>
        {#if job.error}<em>{job.error}</em>{/if}
      </div>
      <div class="job-actions">
        {#if job.output_url}<a href={job.output_url} target="_blank">열기 ↗</a>{/if}
        {#if job.kind === 'recognition' && (job.status === 'failed' || job.status === 'cancelled')}
          <button class="job-retry" disabled={retryingJob === job.id} onclick={() => onRetry(job)}>{retryingJob === job.id ? '재개 중…' : '재개'}</button>
        {:else if (job.kind === 'image' || job.kind === 'video' || job.kind === 'speech') && (job.status === 'failed' || job.status === 'cancelled')}
          <button class="job-retry" disabled={retryingJob === job.id || activeJobs.some((item) => item.kind === 'image' || item.kind === 'video')} onclick={() => onRetry(job)}>{retryingJob === job.id ? '재시도 중…' : '재시도'}</button>
        {/if}
        {#if job.status !== 'queued' && job.status !== 'running'}
          <button class="job-delete" disabled={deletingJob === job.id} onclick={() => onDelete(job)}>삭제</button>
        {/if}
      </div>
    </article>
  {:else}
    <div class="empty">아직 생성 기록이 없습니다.</div>
  {/each}
  <ResultPagination compact label="생성 기록" total={jobs.length} {page} {pageSize} onPageChange={onPage} />
</section>
