<script>
  import { onMount } from 'svelte';
  import { listToolAudits } from '../../api.js';

  export let enabled = true;
  export let onManage = () => {};
  export let onnotify = () => {};

  let audits = [];
  let loading = true;

  const toolNames = {
		skill_view: 'Skill 불러오기', memory_propose: '기억 제안', web_search: '웹 검색', web_fetch: '페이지 읽기', web_collect: '브라우저 수집',
    document_generate: '문서 생성', media_import: '미디어 가져오기', image_generate: '이미지 생성', image_capabilities: '이미지 기능 확인', ssh_exec: 'SSH 실행',
  };
  const decisionNames = { executed: '완료', stored: '저장', automatic: '자동 허용', once: '이번만 허용', conversation: '대화 허용', reject: '거부', execution_error: '실패' };

  onMount(load);

  async function load() {
    loading = true;
    try { audits = await listToolAudits(30); }
    catch (error) { onnotify(error.message, 'error'); }
    finally { loading = false; }
  }

  function when(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? '' : date.toLocaleString();
  }
</script>

<fieldset class="builtin-skills">
  <legend>스킬</legend>
  <label class="check"><input type="checkbox" bind:checked={enabled} /> 필요한 작업 절차 불러오기</label>
  <p><small>라이브러리에서 내장·사용자 스킬을 관리합니다. 변경한 사용 설정은 저장 후 적용됩니다.</small></p>
  <button onclick={onManage}>스킬 관리</button>
</fieldset>

<fieldset>
  <legend>최근 도구 기록</legend>
  <div class="tool-audit-heading"><small>명령 출력과 API 비밀값은 이 기록에 저장하지 않습니다.</small><button onclick={load} disabled={loading}>새로고침</button></div>
  {#if !loading && !audits.length}<small>기록이 없습니다.</small>
  {:else if audits.length}<div class="tool-audit-list">{#each audits as audit}<article><div><strong>{toolNames[audit.tool_name] || audit.tool_name}</strong><span class:error={audit.decision === 'execution_error'}>{decisionNames[audit.decision] || audit.decision}</span></div><small>{when(audit.created_at)}{audit.resource ? ` · ${audit.resource}` : ''}</small>{#if audit.detail}<p>{audit.detail}</p>{/if}</article>{/each}</div>{/if}
</fieldset>
