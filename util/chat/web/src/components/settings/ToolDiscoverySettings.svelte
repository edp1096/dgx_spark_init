<script>
  import { onMount } from 'svelte';
  import { listSkills, listToolAudits } from '../../api.js';

  export let enabled = true;
  export let onnotify = () => {};

  let skills = [];
  let audits = [];
  let loading = true;

  const toolNames = {
		skill_view: 'Skill 불러오기', memory_propose: '기억 제안', web_search: '웹 검색', web_fetch: '페이지 읽기', web_collect: '브라우저 수집',
    media_import: '미디어 가져오기', image_generate: '이미지 생성', image_capabilities: '이미지 기능 확인', ssh_exec: 'SSH 실행',
  };
  const decisionNames = { executed: '완료', stored: '저장', automatic: '자동 허용', once: '이번만 허용', conversation: '대화 허용', reject: '거부', execution_error: '실패' };

  onMount(load);

  async function load() {
    loading = true;
    try { [skills, audits] = await Promise.all([listSkills(), listToolAudits(30)]); }
    catch (error) { onnotify(error.message, 'error'); }
    finally { loading = false; }
  }

  function when(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? '' : date.toLocaleString();
  }
</script>

<fieldset>
  <legend>내장 Skill</legend>
  {#if loading}<small>Skill 목록을 불러오는 중…</small>
  {:else}<div class="skill-catalog">{#each skills as skill}<article class:disabled={!enabled || !skill.available}><div><strong>{skill.name}</strong><span>{enabled && skill.available ? '사용 가능' : '비활성'}</span></div><p>{skill.description}</p><small>{skill.toolsets.join(' · ')}</small></article>{/each}</div>{/if}
  <small>현재 켜진 도구에 맞는 절차만 모델에 노출되고, 전문은 필요할 때만 불러옵니다.</small>
</fieldset>

<fieldset>
  <legend>최근 도구 기록</legend>
  <div class="tool-audit-heading"><small>명령 출력과 API 비밀값은 이 기록에 저장하지 않습니다.</small><button onclick={load} disabled={loading}>새로고침</button></div>
  {#if !loading && !audits.length}<small>기록이 없습니다.</small>
  {:else if audits.length}<div class="tool-audit-list">{#each audits as audit}<article><div><strong>{toolNames[audit.tool_name] || audit.tool_name}</strong><span class:error={audit.decision === 'execution_error'}>{decisionNames[audit.decision] || audit.decision}</span></div><small>{when(audit.created_at)}{audit.resource ? ` · ${audit.resource}` : ''}</small>{#if audit.detail}<p>{audit.detail}</p>{/if}</article>{/each}</div>{/if}
</fieldset>
