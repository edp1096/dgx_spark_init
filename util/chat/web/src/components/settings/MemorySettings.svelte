<script>
  import { onMount } from 'svelte';
  import { createMemory, deleteMemory, listMemories, updateMemory } from '../../api.js';

  export let config;
  export let onnotify = () => {};

  let items = [];
  let loading = true;
  let saving = 0;
  let draft = { kind: 'user', title: '', content: '' };

  onMount(load);

  async function load() {
    loading = true;
    try { items = await listMemories(); }
    catch (error) { onnotify(error.message, 'error'); }
    finally { loading = false; }
  }

  async function add() {
    if (!draft.content.trim() || saving) return;
    saving = -1;
    try {
      const item = await createMemory(draft);
      items = [...items, item];
      draft = { kind: draft.kind, title: '', content: '' };
      onnotify('기억을 추가했습니다.');
    } catch (error) { onnotify(error.message, 'error'); }
    finally { saving = 0; }
  }

  async function save(item) {
    if (!item.content.trim() || saving) return;
    saving = item.id;
    try {
      const updated = await updateMemory(item.id, item);
      items = items.map((entry) => entry.id === item.id ? updated : entry);
      onnotify('기억을 저장했습니다.');
    } catch (error) { onnotify(error.message, 'error'); }
    finally { saving = 0; }
  }

  async function remove(item) {
    if (saving || !confirm('이 기억을 삭제할까요?')) return;
    saving = item.id;
    try {
      await deleteMemory(item.id);
      items = items.filter((entry) => entry.id !== item.id);
      onnotify('기억을 삭제했습니다.');
    } catch (error) { onnotify(error.message, 'error'); }
    finally { saving = 0; }
  }
</script>

<fieldset>
  <legend>회수 설정</legend>
  <label class="check"><input type="checkbox" bind:checked={config.enabled} /> 저장된 기억을 대화에 적용</label>
  <label class="check"><input type="checkbox" bind:checked={config.recall_sessions} disabled={!config.enabled} /> 관련 있는 과거 대화도 검색</label>
  <label class="check"><input type="checkbox" bind:checked={config.allow_proposals} disabled={!config.enabled} /> 모델의 기억 제안 허용 · 저장 전 항상 확인</label>
  <div class="settings-form-row two">
    <label>최대 회수 항목<input type="number" min="1" max="12" bind:value={config.max_results} /></label>
    <label>회수 토큰 예산<input type="number" min="256" max="8192" step="256" bind:value={config.token_budget} /></label>
  </div>
  <small>현재 대화와 실패한 응답은 검색에서 제외합니다. 사용된 항목은 컨텍스트 지도에 표시됩니다.</small>
</fieldset>

<fieldset>
  <legend>새 기억</legend>
  <div class="settings-form-row two">
    <label>분류<select bind:value={draft.kind}><option value="user">사용자 설정</option><option value="memory">장기 기억</option></select></label>
    <label>제목<input bind:value={draft.title} maxlength="120" placeholder="선택 사항" /></label>
  </div>
  <label>내용<textarea rows="3" bind:value={draft.content} maxlength="8000" placeholder="예: 답변은 짧고 간결하게 작성"></textarea></label>
  <button class="memory-add" onclick={add} disabled={saving || !draft.content.trim()}>{saving === -1 ? '추가 중…' : '기억 추가'}</button>
</fieldset>

<fieldset>
  <legend>저장된 기억</legend>
  {#if loading}<small>기억을 불러오는 중…</small>
  {:else if !items.length}<small>저장된 기억이 없습니다.</small>
  {:else}
    <div class="memory-list">
      {#each items as item (item.id)}
        <article class="memory-item" class:disabled={!item.enabled}>
          <div class="memory-item-head">
            <select bind:value={item.kind} aria-label="기억 분류"><option value="user">사용자 설정</option><option value="memory">장기 기억</option></select>
            <input bind:value={item.title} maxlength="120" placeholder="제목" aria-label="기억 제목" />
            <label class="memory-enabled"><input type="checkbox" bind:checked={item.enabled} /> 사용</label>
          </div>
          <textarea rows="3" bind:value={item.content} maxlength="8000" aria-label="기억 내용"></textarea>
          <div class="memory-item-actions"><button onclick={() => remove(item)} disabled={saving}>삭제</button><button class="primary" onclick={() => save(item)} disabled={saving || !item.content.trim()}>{saving === item.id ? '저장 중…' : '저장'}</button></div>
        </article>
      {/each}
    </div>
  {/if}
</fieldset>
