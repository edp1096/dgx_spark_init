<script>
  import { onMount } from 'svelte';
  import { listSkills, getSkill, saveSkill, deleteSkill } from '../api.js';
  export let onnotify = () => {};
  let items = [], search = '', filter = 'all', draft = null, originalName = '', loading = true, saving = false, deleting = false;
  const toolsets = {web:'웹',media:'미디어',image:'이미지 생성',ssh:'SSH',knowledge:'지식',memory:'기억',documents:'문서 생성'};
  $: visible = items.filter(item => (filter === 'all' || (filter === 'builtin' ? item.builtin : !item.builtin)) && `${item.name} ${item.description}`.toLowerCase().includes(search.toLowerCase()));
  onMount(load);
  async function load() {
    loading = true;
    try { items = await listSkills(); } catch(error) { onnotify(error.message,'error'); }
    finally { loading = false; }
  }
  function create() { originalName = ''; deleting = false; draft = {name:'', description:'',instructions:'',toolsets:[],enabled:true}; }
  async function edit(item, copy = false) {
    try {
      const full = await getSkill(item.name);
      originalName = copy ? '' : item.name;
      deleting = false;
      draft = {...full, name:copy ? `${item.name.slice(0,54)}-copy` : item.name, builtin:copy ? false : full.builtin};
    } catch(error) { onnotify(error.message,'error'); }
  }
  async function save() {
    saving = true;
    try {
      if (!originalName && items.some(item => item.name === draft.name)) throw new Error('이미 사용 중인 이름입니다. 다른 이름으로 저장하세요.');
      await saveSkill(draft.name,{description:draft.description,instructions:draft.instructions,toolsets:draft.toolsets,enabled:draft.enabled});
      draft = null; await load(); onnotify('스킬을 저장했습니다. 다음 요청부터 반영됩니다.');
    } catch(error) { onnotify(error.message,'error'); } finally { saving = false; }
  }
  async function toggle(item) {
    saving = true;
    try {
      const data = item.builtin ? {enabled:!item.enabled} : await getSkill(item.name);
      data.enabled = !item.enabled;
      if (!item.builtin) { delete data.name; delete data.builtin; }
      await saveSkill(item.name,data); await load();
    } catch(error) { onnotify(error.message,'error'); } finally {saving = false;}
  }
  async function remove() {
    saving = true;
    try { await deleteSkill(originalName); draft=null; deleting=false; await load();onnotify('스킬을 삭제했습니다.'); }
    catch(error) { onnotify(error.message,'error'); } finally { saving=false; }
  }
</script>

<section class="skill-library">
  {#if draft}
    <form onsubmit={(event) => { event.preventDefault(); save(); }}>
      <div class="skill-toolbar"><h3>{draft.builtin ? '내장 스킬' : originalName ? '스킬 편집' : '스킬 만들기'}</h3><button type="button" onclick={() => draft=null} disabled={saving}>목록으로</button></div>
      {#if draft.builtin}<p>내장 절차입니다. 복사하면 원하는 방식으로 수정할 수 있습니다.</p>{/if}
      <label>이름<input aria-label="스킬 이름" bind:value={draft.name} disabled={!!originalName} required pattern={'[a-z0-9][a-z0-9-]{0,63}'} maxlength="64" placeholder="code-review" /></label>
      <small>영문 소문자·숫자·하이픈. 저장한 이름으로 대화에서 지정합니다.</small>
      <label>사용할 상황<textarea aria-label="사용할 상황" bind:value={draft.description} readonly={draft.builtin} required maxlength="500" rows="2" placeholder="코드 변경사항을 검토하고 오류와 검증 누락을 찾을 때 사용합니다."></textarea></label>
      <label>작업 절차<textarea class="skill-instructions" aria-label="작업 절차" bind:value={draft.instructions} readonly={draft.builtin} required maxlength="16000" rows="12" placeholder="1. 변경 목적과 범위를 확인한다.&#10;2. 관련 코드를 읽는다.&#10;3. 발견한 문제와 검증 결과를 정리한다."></textarea></label>
      <details><summary>상세 설정 · 필요한 도구 {draft.toolsets?.length || 0}개</summary>
        <p>선택한 도구를 사용할 수 있을 때만 이 스킬을 불러옵니다. 도구 없이 쓰는 절차는 비워 두세요.</p>
        <div class="skill-toolsets">{#each Object.entries(toolsets) as [id,label]}<label><input type="checkbox" bind:group={draft.toolsets} value={id} disabled={draft.builtin} />{label}</label>{/each}</div>
      </details>
      {#if !draft.builtin}<label class="skill-check"><input type="checkbox" bind:checked={draft.enabled} />사용</label>{/if}
      <div class="skill-toolbar">
        {#if draft.builtin}<button type="button" onclick={() => edit(draft,true)}>복사해서 수정</button>
        {:else}<button class="primary" type="submit" disabled={saving}>{saving ? '저장 중…' : '저장'}</button>{/if}
        {#if originalName && !draft.builtin}<button type="button" onclick={() => deleting=!deleting} disabled={saving}>삭제</button>{/if}
      </div>
      {#if deleting}<div class="skill-delete" role="alert">이 스킬을 삭제할까요? 기존 대화의 사용 기록은 남습니다.<button type="button" onclick={remove} disabled={saving}>삭제 확인</button><button type="button" onclick={() => deleting=false}>취소</button></div>{/if}
    </form>
  {:else}
    <div class="skill-toolbar"><div><h3>스킬</h3><p>필요한 작업에 불러오는 절차입니다. 관련 요청에서 자동으로 선택하거나 입력창에서 직접 지정합니다.</p></div><button class="primary" onclick={create}>새 스킬</button></div>
    <div class="skill-toolbar"><input aria-label="스킬 검색" placeholder="스킬 검색" bind:value={search} /><select aria-label="스킬 분류" bind:value={filter}><option value="all">전체</option><option value="builtin">내장</option><option value="custom">사용자</option></select></div>
    {#if loading}<p role="status">불러오는 중…</p>{:else if !visible.length}<p>해당하는 스킬이 없습니다.</p>{/if}
    <div class="skill-list">{#each visible as item}<article>
      <div class="skill-toolbar skill-heading"><button class="skill-name" onclick={() => edit(item)}>{item.name}</button><small>{item.builtin ? '내장' : '사용자'}</small><label class="skill-check"><input type="checkbox" checked={item.enabled} onchange={() => toggle(item)} disabled={saving} aria-label={`${item.name} 사용`} />사용</label></div>
      <p>{item.description}</p>
      <div class="skill-toolbar skill-footer"><small>{item.available ? '사용 가능' : item.reason}</small><div><button onclick={() => edit(item)}>{item.builtin ? '절차 보기' : '편집'}</button><button onclick={() => edit(item,true)}>복사</button></div></div>
    </article>{/each}</div>
  {/if}
</section>
<style>
 .skill-library{max-width:960px;margin:0 auto;padding:20px}.skill-toolbar{display:flex;align-items:center;gap:12px;justify-content:space-between;margin:12px 0}.skill-toolbar input{flex:1;min-width:0}.skill-toolbar h3,.skill-toolbar p{margin:0}.skill-list{display:grid;gap:12px}.skill-list article{display:block;margin:0;border:1px solid var(--border,#555);border-radius:10px;padding:12px 16px}.skill-list p{margin:6px 0}.skill-list .skill-toolbar{margin:2px 0;gap:8px}.skill-heading>small{margin-right:auto}.skill-footer>div{display:flex;gap:6px}.skill-footer small{font-size:11px}.skill-list button{padding:5px 8px}.skill-name{font-weight:600;text-align:left}.skill-check,.skill-toolsets label{display:flex;align-items:center;gap:6px}.skill-toolsets{display:flex;gap:16px;flex-wrap:wrap}form>label{display:grid;gap:6px;margin-top:16px}form>label.skill-check{display:flex}textarea,input:not([type=checkbox]){width:100%;box-sizing:border-box}textarea{resize:vertical}small{opacity:.75}.skill-instructions{font-family:monospace}details{margin:16px 0}.skill-delete{padding:12px;border:1px solid var(--border,#555)}.skill-delete button{margin:8px}button{cursor:pointer;border:1px solid #3b4557;border-radius:8px;color:#bdc7d6;background:#1b212b;padding:7px 12px;font-size:12px}button.primary{border-color:#5a7bd0;color:#edf3ff;background:#2a3d69}button:disabled{opacity:.5;cursor:default}.skill-toolbar select{width:130px}.skill-toolbar p{font-size:12px;color:#929cac;margin-top:5px}.skill-check{flex-direction:row}.skill-toolsets label{flex-direction:row}.skill-list article{border-color:#303744;background:#131720}.skill-list p{font-size:13px}.skill-name{border:0;padding-left:0;background:transparent;color:#dfe6f1;font-size:14px}.skill-toolbar h3{font-size:16px}form>small{font-size:11px}details>p{font-size:12px;color:#929cac}:global(html[data-theme="light"]) .skill-list article{background:#fff;border-color:#d8dee8}:global(html[data-theme="light"]) button{color:#42516b;background:#f3f5f8;border-color:#d8dee8}:global(html[data-theme="light"]) button.primary{background:#e3edff;color:#2451a0}:global(html[data-theme="light"]) .skill-name{background:transparent;color:#344054}@media(max-width:600px){.skill-library{padding:12px}.skill-toolbar{flex-wrap:wrap}}
</style>
