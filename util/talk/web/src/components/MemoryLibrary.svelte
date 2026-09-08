<script>
  import { onMount } from 'svelte';
  import { createMemory, deleteMemory, listMemories, updateMemory } from '../api.js';

  export let onnotify = () => {};

  const emptyDraft = () => ({ kind: 'memory', priority: 'preferred', title: '', content: '', enabled: true });
  let items = [];
  let loading = true;
  let saving = 0;
  let query = '';
  let application = 'all';
  let source = 'all';
  let creating = false;
  let editingID = 0;
  let draft = emptyDraft();

  $: filtered = items.filter((item) => (application === 'all' || item.kind === application)
    && (source === 'all' || memorySource(item) === source)
    && (!query.trim() || `${item.title} ${item.content}`.toLowerCase().includes(query.trim().toLowerCase())));

  function memorySource(item) {
    if (item.source_message_id > 0) return 'conversation';
    if (item.source_session_id) return 'proposal';
    return 'manual';
  }

  function sourceLabel(item) {
    switch (memorySource(item)) {
      case 'conversation': return '대화에서 저장';
      case 'proposal': return '모델 제안 승인';
      default: return '직접 작성';
    }
  }

  const applicationLabel = (kind) => kind === 'user' ? '항상 참고' : '관련 있을 때 참고';
  const priorityLabel = (priority) => priority === 'reference' ? '참고' : '우선 적용';

  onMount(load);

  async function load() {
    loading = true;
    try { items = await listMemories(); }
    catch (error) { onnotify(error.message, 'error'); }
    finally { loading = false; }
  }

  function beginCreate() {
    creating = true;
    editingID = 0;
    draft = emptyDraft();
  }

  function beginEdit(item) {
    creating = false;
    editingID = item.id;
    draft = { kind: item.kind, priority: item.priority || 'preferred', title: item.title, content: item.content, enabled: item.enabled };
  }

  function cancelEdit() {
    creating = false;
    editingID = 0;
    draft = emptyDraft();
  }

  async function add() {
    if (!draft.content.trim() || saving) return;
    saving = -1;
    try {
      const item = await createMemory(draft);
      items = [item, ...items];
      cancelEdit();
      onnotify('기억을 추가했습니다.');
    } catch (error) { onnotify(error.message, 'error'); }
    finally { saving = 0; }
  }

  async function save(item) {
    if (!draft.content.trim() || saving) return;
    saving = item.id;
    try {
      const updated = await updateMemory(item.id, draft);
      items = items.map((entry) => entry.id === item.id ? updated : entry);
      cancelEdit();
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
      if (editingID === item.id) cancelEdit();
      onnotify('기억을 삭제했습니다.');
    } catch (error) { onnotify(error.message, 'error'); }
    finally { saving = 0; }
  }
</script>

<div class="library-toolbar memory-library-toolbar">
  <input bind:value={query} placeholder="기억 검색" aria-label="기억 검색" />
  <select bind:value={application} aria-label="호출 범위 필터"><option value="all">모든 호출 범위</option><option value="user">항상 참고</option><option value="memory">관련 있을 때 참고</option></select>
  <select bind:value={source} aria-label="출처 필터"><option value="all">모든 출처</option><option value="manual">직접 작성</option><option value="conversation">대화에서 저장</option><option value="proposal">모델 제안 승인</option></select>
  <button class="primary" onclick={beginCreate} disabled={saving || creating}>＋ 새 기억</button>
</div>

{#if creating}
  <section class="library-card memory-compose-card memory-editor-card">
    <header><div><strong>새 기억</strong><small>행동 규칙은 시스템 프롬프트에, 다시 참고할 사실은 여기에 저장합니다.</small></div></header>
    <div class="settings-form-row three">
      <label>호출 범위<select bind:value={draft.kind}><option value="memory">관련 있을 때 참고</option><option value="user">항상 참고</option></select></label>
      <label>신뢰 수준<select bind:value={draft.priority}><option value="preferred">우선 적용</option><option value="reference">참고</option></select></label>
      <label>제목<input bind:value={draft.title} maxlength="120" placeholder="선택 사항" /></label>
    </div>
    <label>내용<textarea rows="4" bind:value={draft.content} maxlength="8000" placeholder="나중에 다시 참고할 사실을 적으세요."></textarea></label>
    <div class="library-card-actions"><button onclick={cancelEdit} disabled={saving}>취소</button><button class="primary" onclick={add} disabled={saving || !draft.content.trim()}>{saving === -1 ? '추가 중…' : '추가'}</button></div>
  </section>
{/if}

{#if loading}<div class="library-empty">기억을 불러오는 중…</div>
{:else if !filtered.length}<div class="library-empty">조건에 맞는 기억이 없습니다.</div>
{:else}
  <div class="memory-library-list">
    {#each filtered as item (item.id)}
      <article class="library-card memory-library-item" class:disabled={!item.enabled} class:editing={editingID === item.id}>
        {#if editingID === item.id}
          <header><div><strong>{item.title || '제목 없는 기억'}</strong><small>{sourceLabel(item)} · 편집 중</small></div></header>
          <div class="settings-form-row three">
            <label>호출 범위<select bind:value={draft.kind} aria-label="호출 범위"><option value="memory">관련 있을 때 참고</option><option value="user">항상 참고</option></select></label>
            <label>신뢰 수준<select bind:value={draft.priority} aria-label="신뢰 수준"><option value="preferred">우선 적용</option><option value="reference">참고</option></select></label>
            <label>제목<input bind:value={draft.title} maxlength="120" placeholder="선택 사항" aria-label="기억 제목" /></label>
          </div>
          <label>내용<textarea rows="4" bind:value={draft.content} maxlength="8000" aria-label="기억 내용"></textarea></label>
          <footer><label class="memory-enabled"><input type="checkbox" bind:checked={draft.enabled} /> 사용</label><div><button onclick={() => remove(item)} disabled={saving}>삭제</button><button onclick={cancelEdit} disabled={saving}>취소</button><button class="primary" onclick={() => save(item)} disabled={saving || !draft.content.trim()}>{saving === item.id ? '저장 중…' : '저장'}</button></div></footer>
        {:else}
          <header class="memory-card-head">
            <div><strong>{item.title || '제목 없는 기억'}</strong><small>{sourceLabel(item)}{item.updated_at ? ` · ${new Date(item.updated_at).toLocaleString()}` : ''}</small></div>
            <div class="memory-card-badges"><span>{applicationLabel(item.kind)}</span><span class:preferred={(item.priority || 'preferred') === 'preferred'}>{priorityLabel(item.priority)}</span>{#if !item.enabled}<span class="muted">사용 안 함</span>{/if}</div>
          </header>
          <p class="memory-card-content">{item.content}</p>
          <footer><small>호출 범위와 신뢰 수준은 서로 독립적으로 적용됩니다.</small><div><button onclick={() => remove(item)} disabled={saving}>삭제</button><button onclick={() => beginEdit(item)} disabled={saving}>수정</button></div></footer>
        {/if}
      </article>
    {/each}
  </div>
{/if}
