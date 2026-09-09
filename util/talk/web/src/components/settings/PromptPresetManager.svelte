<script>
  import { createClientID } from '../../lib/client-id.js';
  import SettingsHelp from './SettingsHelp.svelte';
  import PromptTextPresets from './PromptTextPresets.svelte';
  import { ensureComposer, promptCategories, selectionGroups, renderPrompt, toggleCondition, switchPromptMode, selectionOf, selectionError, removeBlock } from '../../lib/prompt-composer.js';
  export let model;
  export let onnotify = () => {};
  let c, preview, combinationID = '', editID = '', draft = null;
  $: c = ensureComposer(model);
  $: combinationID = c.combination_id || '';
  $: preview = renderPrompt(c);
  const uid = () => createClientID();
  function update(fn) {
    const next = structuredClone(c); fn(next);
    model = { ...model, prompt_composer: next };
  }
  function changeMode(value) {
    const next = structuredClone(model); switchPromptMode(next, value === 'compose'); model = next;
  }
  function loadCombination(id) {
    const item = c.combinations.find(s => s.id === id);
    update(next => { next.combination_id = id; if (item) Object.assign(next, selectionOf(item)); });
  }
  function saveCombination(replace = false) {
    if (selectionError(c)) return onnotify(selectionError(c), 'error');
    const current = c.combinations.find(s => s.id === combinationID);
    const name = replace && current ? current.name : window.prompt('조합 이름을 입력하세요.');
    if (!name?.trim()) return;
    if (!replace && c.combinations.some(s => s.name === name.trim())) return onnotify('같은 이름의 조합이 있습니다.', 'error');
    const item = { id: replace && current ? current.id : uid(), name: name.trim(), ...selectionOf(c) };
    update(next => { next.combinations = replace && current ? next.combinations.map(s => s.id === item.id ? item : s) : [...next.combinations, item]; next.combination_id = item.id; });
  }
  function renameCombination() {
    const item = c.combinations.find(s => s.id === combinationID);
    const name = window.prompt('조합 이름', item?.name || ''); if (!item || !name?.trim()) return;
    if (c.combinations.some(s => s.id !== item.id && s.name === name.trim())) return onnotify('같은 이름의 조합이 있습니다.', 'error');
    update(next => { next.combinations.find(s => s.id === item.id).name = name.trim(); });
  }
  function deleteCombination() {
    if (!combinationID || !confirm('선택한 조합을 삭제할까요? 현재 선택은 유지됩니다.')) return;
    update(next => { next.combinations = next.combinations.filter(s => s.id !== combinationID); next.combination_id = ''; });
  }
  function editBlock(id) {
    editID = id; draft = id ? structuredClone(c.blocks.find(b => b.id === id)) : null;
  }
  function newBlock(kind) { editID = ''; draft = { id: uid(), kind, category: 'language', group: '', name: '', prompt: '' }; }
  function saveBlock() {
    if (!draft.name.trim() || !draft.prompt.trim()) return onnotify('이름과 내용을 입력하세요.', 'error');
    if (c.blocks.some(b => b.id !== draft.id && b.kind === draft.kind && b.name === draft.name.trim())) return onnotify('같은 이름의 항목이 있습니다.', 'error');
    const item = { ...draft, name: draft.name.trim(), prompt: draft.prompt.trim() };
    if (item.kind === 'persona') { item.category = ''; item.group = ''; }
    else item.category ||= 'language';
    const next = structuredClone(c);
    next.blocks = editID ? next.blocks.map(b => b.id === editID ? item : b) : [...next.blocks, item];
    for (const selection of [next, ...next.combinations]) {
      const error = selectionError(next, selection);
      if (error) return onnotify(`${error} 현재 선택 또는 저장한 조합을 먼저 수정하세요.`, 'error');
    }
    model = { ...model, prompt_composer: next }; editID = item.id; draft = structuredClone(item);
  }
  function deleteBlock() {
    if (!editID || !confirm('이 항목을 삭제하면 현재 선택과 저장한 조합에서도 제외됩니다. 삭제할까요?')) return;
    update(next => removeBlock(next, editID)); editID = ''; draft = null;
  }
</script>

<div class="prompt-composer">
  <fieldset>
    <legend><span>대화 성격과 응답 규칙</span> <SettingsHelp title="프롬프트 작성 방식"><p>텍스트·프리셋은 시스템 프롬프트를 직접 작성하거나 저장한 프리셋을 사용합니다.</p><p>페르소나·조건 조합은 성격·말투·응답 조건을 선택해 하나의 시스템 프롬프트로 적용합니다. 이름과 소개도 함께 적용됩니다.</p></SettingsHelp></legend>
    <label class="settings-field-row"><span>프롬프트 작성 방식</span><select value={c.enabled ? 'compose' : 'direct'} onchange={e => changeMode(e.currentTarget.value)}>
        <option value="direct">텍스트·프리셋</option>
        <option value="compose">페르소나·조건 조합</option>
      </select>
    </label>

  </fieldset>
  {#if !c.enabled}
    <PromptTextPresets bind:model {onnotify} />
  {:else}
    <fieldset>
      <legend>성격과 말투</legend>
      <label class="settings-field-row"><span>페르소나</span><select value={c.persona_id} onchange={e => { const id = e.currentTarget.value; update(next => next.persona_id = id); }}>
          <option value="">없음</option>
          {#each c.blocks.filter(b => b.kind === 'persona') as item}<option value={item.id}>{item.name}</option>{/each}
        </select>
      </label>
      {#if c.persona_id}<p class="persona-description">{c.blocks.find(b => b.id === c.persona_id)?.prompt}</p>{/if}
      <div class="conditions tone-conditions">
        {#each c.blocks.filter(b => b.kind === 'condition' && b.category === 'tone') as item}
          <label title={item.prompt}><input type="checkbox" checked={c.condition_ids.includes(item.id)} onchange={e => { const checked = e.currentTarget.checked; update(next => toggleCondition(next, item.id, checked)); }} />{item.name}</label>
        {/each}
      </div>
    </fieldset>
    <fieldset>
      <legend>응답 조건</legend>
      <small>페르소나를 바꿔도 아래 조건은 유지됩니다.</small>
      <div class="categories">
        {#each promptCategories.filter(category => category.id !== 'tone') as category}
          <details>
            <summary>{category.label} <small>{c.blocks.filter(b => b.category === category.id && c.condition_ids.includes(b.id)).map(b => b.name).join(' · ') || '선택 없음'}</small></summary>
            <div class="conditions">
              {#each c.blocks.filter(b => b.kind === 'condition' && b.category === category.id) as item}
                <label title={item.prompt}><input type="checkbox" checked={c.condition_ids.includes(item.id)} onchange={e => { const checked = e.currentTarget.checked; update(next => toggleCondition(next, item.id, checked)); }} />{item.name}</label>
              {/each}
            </div>
          </details>
        {/each}
      </div>
      <label>직접 추가할 조건<textarea value={c.extra} rows="3" oninput={e => { const text = e.currentTarget.value; update(next => next.extra = text); }}></textarea></label>
    </fieldset>
    <details class="block-editor">
      <summary>조합 저장·관리</summary>
      <small>성격·말투·응답 조건을 한 묶음으로 저장하고 불러옵니다.</small>
      <label class="settings-field-row"><span>저장한 조합</span><select value={combinationID} onchange={e => loadCombination(e.currentTarget.value)}><option value="">현재 선택</option>{#each c.combinations as item}<option value={item.id}>{item.name}</option>{/each}</select></label>
      <div class="actions"><button type="button" onclick={() => saveCombination()}>새 조합 저장</button><button type="button" disabled={!combinationID} onclick={() => saveCombination(true)}>선택 조합 갱신</button><button type="button" disabled={!combinationID} onclick={renameCombination}>조합 이름 변경</button><button type="button" class="danger" disabled={!combinationID} onclick={deleteCombination}>조합 삭제</button></div>
    </details>
    <details class="block-editor">
      <summary>페르소나·조건 편집</summary>
      <label class="settings-field-row"><span>편집할 항목</span><select value={editID} onchange={e => editBlock(e.currentTarget.value)}><option value="">항목 선택</option>{#each c.blocks as item}<option value={item.id}>{item.kind === 'persona' ? '페르소나' : '조건'} · {item.name}</option>{/each}</select></label>
      <div class="actions"><button type="button" onclick={() => newBlock('persona')}>페르소나 추가</button><button type="button" onclick={() => newBlock('condition')}>조건 추가</button></div>
      {#if draft}
        <label class="settings-field-row"><span>항목 이름</span><input bind:value={draft.name} /></label>
        <label class="settings-field-row"><span>항목 종류</span><select bind:value={draft.kind} onchange={() => { if (draft.kind === 'condition') draft.category ||= 'language'; }}><option value="persona">페르소나</option><option value="condition">추가 조건</option></select></label>
        {#if draft.kind === 'condition'}<label class="settings-field-row"><span>조건 분류</span><select bind:value={draft.category}>{#each promptCategories as category}<option value={category.id}>{category.label}</option>{/each}</select></label><label class="settings-field-row"><span>선택 제한</span><select bind:value={draft.group}>{#each selectionGroups as [id, label]}<option value={id}>{label}</option>{/each}</select></label>{/if}
        <label>항목 내용<textarea bind:value={draft.prompt} rows="5"></textarea></label>
        <div class="actions"><button type="button" onclick={saveBlock}>항목 저장</button><button type="button" class="danger" disabled={!editID} onclick={deleteBlock}>항목 삭제</button></div>
      {/if}
    </details>
    <fieldset>
      <legend>적용될 프롬프트</legend>
      <label>최종 프롬프트 미리보기<textarea readonly value={preview} rows="6"></textarea></label>
      <small>{Array.from(preview).length}자 · 페르소나 {c.persona_id ? 1 : 0}개 · 추가 조건 {c.condition_ids.length}개</small>
    </fieldset>
  {/if}
</div>

<style>
  .prompt-composer { display: grid; gap: 12px; min-width: 0; }
  fieldset, .block-editor { display: grid; gap: 12px; min-width: 0; }
  label { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
  select, textarea, input { max-width: 100%; box-sizing: border-box; }
  textarea { width: 100%; resize: vertical; }
  .actions { display: flex; flex-wrap: wrap; gap: 6px; }
  .categories { display: grid; gap: 8px; }
  details { border: 1px solid #80808040; border-radius: 8px; padding: 10px; }
  summary { cursor: pointer; }
  .persona-description { margin: 0; padding: 10px; border-radius: 8px; background: #80808012; white-space: pre-wrap; font-size: 13px; max-height: 130px; overflow: auto; }
  .tone-conditions { padding-top: 0; }
  summary small { margin-left: 8px; opacity: .65; }
  .conditions { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 190px), 1fr)); gap: 10px; padding-top: 12px; }
  .conditions label { flex-direction: row; align-items: center; }
  .conditions input { width: auto; margin: 0; }
  .block-editor[open] > label, .block-editor[open] > .actions { margin-top: 12px; }
</style>
