<script>
  import { onMount } from 'svelte';
  import { searchConversationPage } from '../api.js';

  export let initialQuery = '';
  export let onclose = () => {};
  export let onselect = () => {};

  let query = initialQuery;
  let sort = 'relevance';
  let scope = 'all';
  let dateFrom = '';
  let dateTo = '';
  let items = [];
  let cursor = '';
  let loading = false;
  let error = '';
  let sequence = 0;
  let input;

  onMount(() => {
    input?.focus();
    search(true);
    const escape = (event) => { if (event.key === 'Escape') onclose(); };
    document.addEventListener('keydown', escape);
    return () => document.removeEventListener('keydown', escape);
  });

  async function search(reset) {
    const term = query.trim();
    if (!term || loading) return;
    const requestSequence = ++sequence;
    const nextCursor = reset ? '' : cursor;
    if (reset) { items = []; cursor = ''; }
    loading = true;
    error = '';
    try {
      const page = await searchConversationPage(term, { limit: 20, sort, scope, from: dateFrom, to: dateTo, cursor: nextCursor });
      if (requestSequence !== sequence) return;
      items = mergeConversations(reset ? [] : items, page.items || []);
      cursor = page.next_cursor || '';
    } catch (searchError) {
      if (requestSequence === sequence) error = searchError.message;
    } finally {
      if (requestSequence === sequence) loading = false;
    }
  }

  function restart() {
    sequence++;
    loading = false;
    search(true);
  }

  function choose(item) {
    onselect(item);
  }

  function mergeConversations(current, incoming) {
    const seen = new Set(current.map((item) => item.session_id));
    const merged = [...current];
    for (const item of incoming) {
      if (!item.session_id || seen.has(item.session_id)) continue;
      seen.add(item.session_id);
      merged.push(item);
    }
    return merged;
  }

  function escapeHTML(value) {
    return String(value || '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
  }

  function highlighted(value) {
    let safe = escapeHTML(value);
    const terms = [...new Set(query.trim().split(/\s+/).filter(Boolean))]
      .sort((left, right) => right.length - left.length)
      .map((term) => escapeHTML(term).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    if (!terms.length) return safe;
    return safe.replace(new RegExp(`(${terms.join('|')})`, 'gi'), '<mark>$1</mark>');
  }

  function dateLabel(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? '' : date.toLocaleString();
  }
</script>

<div class="modal-backdrop search-modal-backdrop" role="presentation" onclick={(event) => event.target === event.currentTarget && onclose()}>
  <div class="search-modal" role="dialog" aria-modal="true" aria-labelledby="search-modal-title">
    <header class="search-modal-title"><div><h2 id="search-modal-title">전체 대화 검색</h2><small>전체 개수는 계산하지 않고 20개씩 불러옵니다.</small></div><button onclick={onclose} aria-label="검색 닫기">×</button></header>
    <form class="search-modal-form" onsubmit={(event) => { event.preventDefault(); restart(); }}>
      <div class="search-modal-input"><span aria-hidden="true">⌕</span><input bind:this={input} bind:value={query} maxlength="200" aria-label="검색어" placeholder="대화 제목이나 메시지 검색" /><button type="submit" disabled={!query.trim() || loading}>검색</button></div>
      <div class="search-modal-filters">
        <label>정렬<select bind:value={sort} onchange={restart}><option value="relevance">관련도순</option><option value="recent">최신순</option></select></label>
        <label>범위<select bind:value={scope} onchange={restart}><option value="all">제목과 본문</option><option value="title">제목만</option><option value="content">본문만</option></select></label>
        <label>시작일<input type="date" bind:value={dateFrom} onchange={restart} /></label>
        <label>종료일<input type="date" bind:value={dateTo} onchange={restart} /></label>
      </div>
    </form>
    <div class="search-modal-results" aria-live="polite">
      {#if error}<div class="search-modal-state error">{error}</div>
      {:else if !items.length && loading}<div class="search-modal-state">검색 중…</div>
      {:else if !items.length}<div class="search-modal-state">일치하는 메시지가 없습니다.</div>
      {:else}
        {#each items as item (item.session_id)}
          <button class="search-result-card" onclick={() => choose(item)}>
            <div><strong>{@html highlighted(item.title)}</strong><span>{item.message_id ? `${item.role === 'user' ? '나' : 'AI'} · ${dateLabel(item.created_at)}` : '빈 대화'}</span></div>
            {#if item.content}<p>{@html highlighted(item.content)}</p>{/if}
          </button>
        {/each}
      {/if}
    </div>
    <footer class="search-modal-footer"><span>{items.length ? `${items.length}개 대화 표시` : ''}</span>{#if cursor}<button onclick={() => search(false)} disabled={loading}>{loading ? '불러오는 중…' : '다음 20개 탐색'}</button>{/if}</footer>
  </div>
</div>
