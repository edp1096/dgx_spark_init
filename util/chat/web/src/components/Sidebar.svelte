<script>
  import { onMount } from 'svelte';
  import Avatar from './Avatar.svelte';
  import { searchConversations } from '../api.js';

  export let groups = [];
  export let sessionsByGroup = {};
  export let ungroupedSessions = [];
  export let collapsedGroups = {};
  export let activeId = '';
  export let sessionRuns = {};
  export let assistantAvatar = 'preset:spark';
  export let onclose = () => {};
  export let onAddSession = () => {};
  export let onAddGroup = () => {};
  export let onToggleGroup = () => {};
  export let onEditGroup = () => {};
  export let onReorderGroup = () => {};
  export let onRemoveGroup = () => {};
  export let onSelect = () => {};
  export let onChangeSessionGroup = () => {};
  export let onRemoveSession = () => {};
  export let onOpenSettings = () => {};
  export let onOpenLibrary = () => {};
  export let libraryOpen = false;
  export let onStartResize = () => {};
  export let onSearchResult = () => {};
  export let onSearchMore = () => {};

  let sessionMenuId = '';
  let searchQuery = '';
  let searchResults = [];
  let searchLoading = false;
  let searchError = '';
  let searchTimer;
  let searchSequence = 0;

  onMount(() => {
    function closeOnOutsidePointer(event) {
      if (sessionMenuId && !event.target.closest?.('.session-menu, .session-more')) sessionMenuId = '';
      if (searchQuery && !event.target.closest?.('.conversation-search')) clearSearch();
    }

    function closeOnEscape(event) {
      if (sessionMenuId && event.key === 'Escape') sessionMenuId = '';
    }

    document.addEventListener('pointerdown', closeOnOutsidePointer, true);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      clearTimeout(searchTimer);
      document.removeEventListener('pointerdown', closeOnOutsidePointer, true);
      document.removeEventListener('keydown', closeOnEscape);
    };
  });

  function queueSearch() {
    clearTimeout(searchTimer);
    searchError = '';
    const query = searchQuery.trim();
    if (!query) { searchResults = []; searchLoading = false; return; }
    searchLoading = true;
    const sequence = ++searchSequence;
    searchTimer = setTimeout(async () => {
      try {
        const results = await searchConversations(query, 5);
        if (sequence === searchSequence) searchResults = results;
      } catch (error) {
        if (sequence === searchSequence) searchError = error.message;
      } finally {
        if (sequence === searchSequence) searchLoading = false;
      }
    }, 180);
  }

  function clearSearch() {
    clearTimeout(searchTimer);
    searchSequence++;
    searchQuery = '';
    searchResults = [];
    searchLoading = false;
    searchError = '';
  }

  function chooseSearchResult(item) {
    clearSearch();
    onSearchResult(item);
  }

  function showAllSearchResults() {
    const query = searchQuery.trim();
    if (!query) return;
    clearSearch();
    onSearchMore(query);
  }

  function toggleSessionMenu(id) {
    sessionMenuId = sessionMenuId === id ? '' : id;
  }

  function changeSessionGroup(session, groupId) {
    sessionMenuId = '';
    onChangeSessionGroup(session, groupId);
  }

  function removeSession(id) {
    sessionMenuId = '';
    onRemoveSession(id);
  }
</script>

<aside class="sidebar">
  <div class="brand"><span class="mark"><Avatar value={assistantAvatar} alt="SparkTalk" /></span><strong>SparkTalk</strong><button class="sidebar-close" onclick={onclose} aria-label="사이드바 닫기">×</button></div>
  <div class="sidebar-actions">
    <button class="new-chat" onclick={onAddSession}>＋ 새 대화</button>
    <button class="new-group" onclick={onAddGroup} title="그룹 만들기" aria-label="그룹 만들기">＋ 폴더</button>
  </div>
  <div class="conversation-search">
    <span aria-hidden="true">⌕</span><input bind:value={searchQuery} oninput={queueSearch} onkeydown={(event) => { if (event.key === 'Escape') clearSearch(); }} placeholder="전체 대화 검색" aria-label="전체 대화 검색" />
    {#if searchQuery}<button onclick={clearSearch} aria-label="검색 지우기">×</button>{/if}
    {#if searchQuery}
      <div class="conversation-search-results">
        {#if searchLoading}<small>검색 중…</small>
        {:else if searchError}<small class="search-error">{searchError}</small>
        {:else if !searchResults.length}<small>일치하는 대화가 없습니다.</small>
        {:else}{#each searchResults as item}<button onclick={() => chooseSearchResult(item)}><strong>{item.title}</strong><span>{item.message_id ? `${item.role === 'user' ? '나' : 'AI'} · ${item.content}` : '빈 대화'}</span></button>{/each}{/if}
        {#if !searchLoading && !searchError}<button class="conversation-search-more" onclick={showAllSearchResults}>검색 결과 더보기 →</button>{/if}
      </div>
    {/if}
  </div>
  <nav>
    {#each groups as group, groupIndex}
      <section class="chat-group">
        <div class="group-heading">
          <button class="group-toggle" onclick={() => onToggleGroup(group.id)} aria-expanded={!collapsedGroups[group.id]}>
            <span>{collapsedGroups[group.id] ? '▸' : '▾'} 📁 {group.name}</span><small>{(sessionsByGroup[group.id] || []).length}</small>
          </button>
          <div class="group-actions">
            <button onclick={() => onReorderGroup(group, 'up')} disabled={groupIndex === 0} title="위로 이동">↑</button>
            <button onclick={() => onReorderGroup(group, 'down')} disabled={groupIndex === groups.length - 1} title="아래로 이동">↓</button>
            <button onclick={() => onEditGroup(group)} title="이름 변경">✎</button>
            <button class="danger" onclick={() => onRemoveGroup(group)} title="그룹 삭제">×</button>
          </div>
        </div>
        {#if !collapsedGroups[group.id]}
          {#each sessionsByGroup[group.id] || [] as session}
            <div class="session-row" class:active={session.id === activeId} class:generating={Boolean(sessionRuns[session.id])}>
              <button class="session-select" onclick={() => onSelect(session.id)}>{session.title}</button>
              {#if sessionRuns[session.id]}<span class="session-running" title="답변 생성 중" aria-label="답변 생성 중">●</span>{/if}
              <button class="session-more" onclick={() => toggleSessionMenu(session.id)} aria-label={`${session.title} 메뉴`} aria-haspopup="menu" aria-expanded={sessionMenuId === session.id}>⋯</button>
              {#if sessionMenuId === session.id}
                <div class="session-menu" role="menu">
                  <strong>그룹 이동</strong>
                  <button onclick={() => changeSessionGroup(session, '')}>그룹 없음</button>
                  {#each groups as target}<button class:current={target.id === session.group_id} onclick={() => changeSessionGroup(session, target.id)}>▸ {target.name}</button>{/each}
                  <hr /><button class="danger" onclick={() => removeSession(session.id)} disabled={Boolean(sessionRuns[session.id])}>대화 삭제</button>
                </div>
              {/if}
            </div>
          {/each}
        {/if}
      </section>
    {/each}
    <section class="chat-group ungrouped">
      <button class="group-toggle" onclick={() => onToggleGroup('__ungrouped__')} aria-expanded={!collapsedGroups.__ungrouped__}>
        <span>{collapsedGroups.__ungrouped__ ? '▸' : '▾'} 대화</span><small>{ungroupedSessions.length}</small>
      </button>
      {#if !collapsedGroups.__ungrouped__}
        {#each ungroupedSessions as session}
          <div class="session-row" class:active={session.id === activeId} class:generating={Boolean(sessionRuns[session.id])}>
            <button class="session-select" onclick={() => onSelect(session.id)}>{session.title}</button>
            {#if sessionRuns[session.id]}<span class="session-running" title="답변 생성 중" aria-label="답변 생성 중">●</span>{/if}
            <button class="session-more" onclick={() => toggleSessionMenu(session.id)} aria-label={`${session.title} 메뉴`} aria-haspopup="menu" aria-expanded={sessionMenuId === session.id}>⋯</button>
            {#if sessionMenuId === session.id}
              <div class="session-menu" role="menu">
                <strong>그룹 이동</strong>
                <button class:current={!session.group_id} onclick={() => changeSessionGroup(session, '')}>그룹 없음</button>
                {#each groups as group}<button onclick={() => changeSessionGroup(session, group.id)}>▸ {group.name}</button>{/each}
                <hr /><button class="danger" onclick={() => removeSession(session.id)} disabled={Boolean(sessionRuns[session.id])}>대화 삭제</button>
              </div>
            {/if}
          </div>
        {/each}
      {/if}
    </section>
  </nav>
  <div class="sidebar-footer-actions">
    <button class="library-button" class:active={libraryOpen} onclick={onOpenLibrary}>▤ 기억·지식</button>
    <button class="settings-button" onclick={onOpenSettings}>⚙ 설정</button>
  </div>
  <button class="resize-handle" onpointerdown={onStartResize} aria-label="사이드바 폭 조절"></button>
</aside>
<button class="sidebar-backdrop" onclick={onclose} aria-label="사이드바 닫기"></button>
