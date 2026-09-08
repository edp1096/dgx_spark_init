<script>
  import { onDestroy, onMount } from 'svelte';
  import { listKnowledgeJobs, updateKnowledgeJob } from '../api.js';

  export let onnotify = () => {};
  export let refreshKey = 0;

  let jobs = [];
  let loading = true;
  let busy = '';
  let timer;
  let lastRefreshKey = refreshKey;

  $: if (refreshKey !== lastRefreshKey) { lastRefreshKey = refreshKey; load(); }

  onMount(() => {
    load();
    timer = setInterval(() => {
      if (jobs.some((job) => ['queued', 'running'].includes(job.status))) load(false);
    }, 1800);
  });
  onDestroy(() => clearInterval(timer));

  async function load(showLoading = true) {
    if (showLoading) loading = true;
    try { jobs = await listKnowledgeJobs(); }
    catch (error) { onnotify(error.message, 'error'); }
    finally { loading = false; }
  }

  async function act(job, action) {
    if (busy) return;
    busy = job.id;
    try {
      await updateKnowledgeJob(job.id, action);
      await load(false);
    } catch (error) { onnotify(error.message, 'error'); }
    finally { busy = ''; }
  }

  function label(status) {
    return ({ paused: '시작 대기', queued: '대기 중', running: '가져오는 중', completed: '완료', completed_with_errors: '일부 실패', failed: '실패', canceled: '취소됨' })[status] || status;
  }

  function percent(job) {
    return job.total_items ? Math.round(((job.completed_items + job.failed_items) / job.total_items) * 100) : 0;
  }
</script>

{#if loading && !jobs.length}<div class="library-empty">가져오기 작업을 불러오는 중…</div>
{:else if !jobs.length}<div class="library-empty">가져오기 작업이 없습니다.</div>
{:else}
  <div class="knowledge-job-list">
    {#each jobs as job (job.id)}
      <article class="library-card knowledge-job" class:running={job.status === 'running'}>
        <header><div><strong>{job.title || '전자책 가져오기'}</strong><small>{job.adapter || '웹 자료'} · {job.total_items}쪽</small></div><span class="job-status">{label(job.status)}</span></header>
        <div class="job-progress"><span style:width={`${percent(job)}%`}></span></div>
        <div class="job-meta"><span>{job.completed_items}/{job.total_items} 완료</span>{#if job.failed_items}<span class="knowledge-error">{job.failed_items} 실패</span>{/if}{#if job.status === 'running'}<span>{job.current_item}쪽 처리 중</span>{/if}</div>
        {#if job.error}<small class="knowledge-error">{job.error}</small>{/if}
        {#if job.failures?.length}<details class="job-failures"><summary>실패한 쪽 확인</summary><div>{#each job.failures as failure}<p><b>{failure.ordinal}쪽</b><span>{failure.error || '가져오지 못했습니다.'}</span></p>{/each}</div></details>{/if}
        <footer><a href={job.source_url} target="_blank" rel="noreferrer">원본 페이지</a><div>
          {#if ['paused', 'canceled'].includes(job.status)}<button class="primary" onclick={() => act(job, 'resume')} disabled={busy}>가져오기 시작</button>
          {:else if ['queued', 'running'].includes(job.status)}<button onclick={() => act(job, 'pause')} disabled={busy}>일시중지</button><button onclick={() => act(job, 'cancel')} disabled={busy}>취소</button>
          {:else if ['failed', 'completed_with_errors'].includes(job.status)}<button class="primary" onclick={() => act(job, 'retry')} disabled={busy}>실패 재시도</button>{/if}
        </div></footer>
      </article>
    {/each}
  </div>
{/if}
