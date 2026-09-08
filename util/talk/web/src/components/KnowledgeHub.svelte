<script>
  import { onDestroy } from 'svelte';
  import KnowledgeSettings from './settings/KnowledgeSettings.svelte';
  import MemoryLibrary from './MemoryLibrary.svelte';
  import SkillLibrary from './SkillLibrary.svelte';
  import WorkflowLibrary from './WorkflowLibrary.svelte';
  import KnowledgeJobs from './KnowledgeJobs.svelte';

  export let health = null;
  export let initialTab = 'memory';
  export let onclose = () => {};

  let activeTab = initialTab;
  let helpOpen = false;
  let helpDialog;

  function toggleHelp() {
    helpDialog.showModal();
    helpDialog.scrollTop = 0;
    helpOpen = true;
  }
  let notice = null;
  let noticeTimer;
  let jobRefreshKey = 0;
  let knowledgeTab = 'sources';

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
    activeTab = 'knowledge'; knowledgeTab = 'jobs';
  }
</script>

<section class="knowledge-hub">
  <header class="knowledge-hub-header"><div><button class="knowledge-hub-back" onclick={onclose} aria-label="대화로 돌아가기">←</button><div><strong>라이브러리</strong><small>기억할 사실 · 참고할 자료 · 작업 절차</small></div></div><button class="library-help-button" onclick={toggleHelp} aria-label="라이브러리 도움말" aria-haspopup="dialog" aria-expanded={helpOpen} aria-controls="library-help" title="라이브러리 도움말"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><circle cx="12" cy="7.5" r=".8" fill="currentColor" stroke="none"/></svg><span>도움말</span></button></header>
  <nav class="knowledge-hub-tabs" aria-label="라이브러리 분류">
    <button class:active={activeTab === 'memory'} onclick={() => activeTab = 'memory'}>기억</button>
    <button class:active={activeTab === 'knowledge'} onclick={() => activeTab = 'knowledge'}>지식</button>
    <button class:active={activeTab === 'workflows'} onclick={() => activeTab = 'workflows'}>작업 절차</button>
    <button class:active={activeTab === 'skills'} onclick={() => activeTab = 'skills'}>스킬</button>
  </nav>
  <div class="knowledge-hub-body">
    {#if activeTab === 'memory'}<MemoryLibrary onnotify={notify} />
    {:else if activeTab === 'knowledge'}
      <nav class="knowledge-hub-tabs" aria-label="지식 보기"><button class:active={knowledgeTab === 'sources'} onclick={() => knowledgeTab='sources'}>자료실</button><button class:active={knowledgeTab === 'jobs'} onclick={() => knowledgeTab='jobs'}>가져오기 작업</button></nav>
      {#if knowledgeTab === 'sources'}<KnowledgeSettings onnotify={notify} health={health?.extra?.collector} onjobcreated={jobCreated} />{:else}<KnowledgeJobs onnotify={notify} refreshKey={jobRefreshKey} />{/if}
    {:else if activeTab === 'workflows'}<WorkflowLibrary onnotify={notify} />
    {:else}<SkillLibrary onnotify={notify} />{/if}
  </div>
      <dialog bind:this={helpDialog} onclose={()=>helpOpen=false} onclick={event=>{if(event.target!==helpDialog)return;const r=helpDialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)helpDialog.close();}} id="library-help" class="library-help" aria-labelledby="library-help-title">
        <div class="library-help-heading"><h2 id="library-help-title">라이브러리 사용 안내</h2><button class="library-help-button" onclick={()=>helpDialog.close()} aria-label="도움말 닫기">×</button></div>
        <dl class="library-help-roles">
          <div><dt>기억</dt><dd>취향·환경 등 다음 대화에도 기억할 사실</dd></div>
          <div><dt>지식</dt><dd>답변할 때 찾아볼 문서와 참고 자료</dd></div>
          <div><dt>스킬</dt><dd>코드 리뷰·자료 조사 같은 개별 작업 지침</dd></div>
          <div><dt>작업 절차</dt><dd>여러 스킬을 순서대로 묶은 단계별 작업</dd></div>
        </dl>
        <h3>작업 절차는 이렇게 사용하세요</h3>
        <ol>
          <li><strong>구성하기</strong> — 코딩·서버·조사·미디어·문서·이미지 등 기본 절차를 복사해 순서, 완료 조건, 검증·재시도를 편집합니다.</li>
          <li><strong>적용하기</strong> — 대화 입력창의 <strong>＋ → 스킬·작업 절차</strong>에서 순서를 미리 보고 적용합니다. 지정하지 않으면 모델이 필요한 스킬이나 절차를 선택합니다.</li>
          <li><strong>확인·이어가기</strong> — 대화 아래 <strong>작업 진행</strong>에서 결과와 실행 근거를 확인하고, 중단한 작업은 <strong>이 작업 이어하기</strong>로 재개합니다.</li>
        </ol>
        <p>현재 단계의 스킬과 이전 결과를 이어받습니다. 실행 검증이 필요한 단계는 실제 도구 성공 근거를 확인하며, 편집한 구성은 새 작업부터 적용됩니다.</p>
        <p class="library-help-note">전역 시스템 프롬프트는 모든 대화의 기본 지침입니다. 스킬·작업 절차를 선택해도 도구 권한과 승인 설정은 유지됩니다.</p>
      </dialog>
  {#if notice}
		<div class="library-toast" class:error={notice.kind === 'error'} role={notice.kind === 'error' ? 'alert' : 'status'}>
			<span>{notice.message}</span>
			<div class="library-toast-actions"><button onclick={copyNotice}>{notice.copied ? '복사됨' : '복사'}</button><button onclick={() => notice = null} aria-label="알림 닫기">×</button></div>
		</div>
	{/if}
</section>

<style>
  .knowledge-hub-header { display:flex; align-items:center; justify-content:space-between; gap:12px; }
  .library-help-button { display:inline-flex; align-items:center; justify-content:center; gap:6px; flex-shrink:0; min-width:34px; min-height:34px; padding:5px 9px; border:1px solid #8994a455; border-radius:8px; background:transparent; color:inherit; cursor:pointer; font-size:12px; }
  .library-help-button:hover { background:#8994a41a; }
  .library-help { box-sizing:border-box; border:1px solid #8994a455; border-radius:12px; padding:18px; margin:auto; width:min(680px, calc(100vw - 32px)); max-height:calc(100dvh - 32px); overflow-y:auto; overscroll-behavior:contain; background:#171b24; color:#e8eaf0; box-shadow:0 18px 60px #0006; font-size:13px; line-height:1.65; }
  .library-help::backdrop { background:#0008; }
  :global(html[data-theme="light"]) .library-help { background:#fff; color:#20242d; }
  .library-help-heading { display:flex; align-items:center; justify-content:space-between; gap:12px; }
  .library-help h2 { margin:0; font-size:15px; }
  .library-help h3 { margin:16px 0 8px; font-size:13px; }
  .library-help-roles { display:grid; grid-template-columns:1fr 1fr; gap:12px 20px; margin:14px 0; }
  .library-help dt { font-weight:600; }
  .library-help dd { margin:2px 0 0; opacity:.8; }
  .library-help ol { padding-left:20px; margin:0; }
  .library-help li + li { margin-top:8px; }
  .library-help p { margin:14px 0 0; }
  .library-help-note { padding-top:12px; border-top:1px solid #8994a433; opacity:.75; font-size:12px; }
  @media(max-width:600px) { .library-help { padding:14px; } .library-help-roles { grid-template-columns:1fr; gap:10px; } }
</style>
