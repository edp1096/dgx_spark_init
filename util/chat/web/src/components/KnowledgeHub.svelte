<script>
  import { onDestroy } from 'svelte';
  import KnowledgeSettings from './settings/KnowledgeSettings.svelte';
  import MemoryLibrary from './MemoryLibrary.svelte';
  import KnowledgeJobs from './KnowledgeJobs.svelte';

  export let health = null;
  export let initialTab = 'memory';
  export let onclose = () => {};

  let activeTab = initialTab;
  let notice = null;
  let noticeTimer;
  let jobRefreshKey = 0;

  onDestroy(() => clearTimeout(noticeTimer));

  function notify(message, kind = 'success') {
    clearTimeout(noticeTimer);
    notice = { message, kind };
		if (kind !== 'error') noticeTimer = setTimeout(() => { notice = null; }, 4000);
  }

	async function copyNotice() {
		if (!notice?.message) return;
		try {
			await navigator.clipboard.writeText(notice.message);
		} catch {
			const input = document.createElement('textarea');
			input.value = notice.message;
			input.style.position = 'fixed';
			input.style.opacity = '0';
			document.body.appendChild(input);
			input.select();
			document.execCommand('copy');
			input.remove();
		}
		notice = { ...notice, copied: true };
	}

  function jobCreated() {
    jobRefreshKey += 1;
    activeTab = 'jobs';
  }
</script>

<section class="knowledge-hub">
  <header class="knowledge-hub-header"><div><button class="knowledge-hub-back" onclick={onclose} aria-label="대화로 돌아가기">←</button><div><strong>기억·지식</strong><small>짧은 기억과 출처가 있는 자료를 분리해서 관리합니다.</small></div></div></header>
  <nav class="knowledge-hub-tabs" aria-label="기억과 지식 관리">
    <button class:active={activeTab === 'memory'} onclick={() => activeTab = 'memory'}>기억</button>
    <button class:active={activeTab === 'knowledge'} onclick={() => activeTab = 'knowledge'}>지식 자료실</button>
    <button class:active={activeTab === 'jobs'} onclick={() => activeTab = 'jobs'}>가져오기 작업</button>
  </nav>
  <div class="knowledge-hub-body">
    {#if activeTab === 'memory'}<MemoryLibrary onnotify={notify} />
    {:else if activeTab === 'knowledge'}<KnowledgeSettings onnotify={notify} health={health?.extra?.collector} onjobcreated={jobCreated} />
    {:else}<KnowledgeJobs onnotify={notify} refreshKey={jobRefreshKey} />{/if}
  </div>
  {#if notice}
		<div class="library-toast" class:error={notice.kind === 'error'} role={notice.kind === 'error' ? 'alert' : 'status'}>
			<span>{notice.message}</span>
			<div class="library-toast-actions"><button onclick={copyNotice}>{notice.copied ? '복사됨' : '복사'}</button><button onclick={() => notice = null} aria-label="알림 닫기">×</button></div>
		</div>
	{/if}
</section>
