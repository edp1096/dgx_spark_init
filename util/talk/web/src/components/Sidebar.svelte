<script>
  import { onMount } from 'svelte';
  import Avatar from './Avatar.svelte';
  import SessionPager from './SessionPager.svelte';
  import { sessionPage, SESSION_PAGE_SIZE } from '../lib/session-pages.js';
  import { searchConversations } from '../api.js';

  export let groups = [];
  export let sessionsByGroup = {};
  export let ungroupedSessions = [];
  export let collapsedGroups = {};
  export let foldersCollapsed = false;
  export let activeId = '';
  export let sessionRuns = {};
  export let assistantName = 'SparkTalk';
  export let assistantAvatar = 'preset:spark';
  export let onclose = () => {};
  export let onAddSession = () => {};
  export let onAddGroup = () => {};
  export let onToggleGroup = () => {};
  export let onToggleFolders = () => {};
  export let onEditGroup = () => {};
  export let onReorderGroup = () => {};
  export let onRemoveGroup = () => {};
  export let onSelect = () => {};
  export let onChangeSessionGroup = () => {};
  export let onRemoveSession = () => {};
  export let onOpenSettings = () => {};
  export let onOpenProfile = () => {};
  export let onOpenLibrary = () => {};
  export let libraryOpen = false;
  export let onStartResize = () => {};
  export let onSearchResult = () => {};
  export let onSearchMore = () => {};

  export let onRemoveSelected = async () => false;
  let selecting = false;
  let selectedIds = [];
  let deleting = false;
  $: selectable = [...ungroupedSessions, ...Object.values(sessionsByGroup).flat()].filter(s => !sessionRuns[s.id]);
  $: selectedIds = selectedIds.filter(id => selectable.some(s => s.id === id));
  function toggleSelection(id) {
    if (deleting || sessionRuns[id]) return;
    selectedIds = selectedIds.includes(id) ? selectedIds.filter(x => x !== id) : [...selectedIds, id];
  }
  async function deleteSelected() {
    deleting = true;
    try { if (await onRemoveSelected(selectedIds)) { selectedIds = []; selecting = false; } }
    finally { deleting = false; }
  }
  let sessionMenuId = '';
  let sessionPages = {};
  let followedActive = '';
  $: ungroupedPage = sessionPage(ungroupedSessions, sessionPages.__ungrouped__);
  $: groupPages = Object.fromEntries(Object.entries(sessionsByGroup).map(([id, items]) => [id, sessionPage(items, sessionPages[id])]));
  $: followActive(activeId, sessionsByGroup, ungroupedSessions);

  function changePage(key, page) {
    sessionMenuId = '';
    sessionPages = { ...sessionPages, [key]: page };
  }

  // Selecting an older conversation through search reveals its page, without
  // preventing the user from browsing other pages while that chat stays open.
  function followActive(id, grouped, ungrouped) {
    if (!id) { followedActive = ''; return; }
    for (const [key, items] of [['__ungrouped__', ungrouped], ...Object.entries(grouped)]) {
      const index = items.findIndex(item => item.id === id);
      if (index < 0) continue;
      const identity = `${key}:${id}`;
      if (identity !== followedActive) {
        followedActive = identity;
        changePage(key, Math.floor(index / SESSION_PAGE_SIZE));
      }
      return;
    }
  }

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
  <div class="brand"><button type="button" class="mark profile-avatar" aria-label="AI 캐릭터 설정" title="AI 캐릭터 설정" onclick={onOpenProfile}><Avatar value={assistantAvatar} alt={assistantName} /></button><strong class="character-name" title={assistantName}>{assistantName}</strong><button class="sidebar-close" onclick={onclose} aria-label="사이드바 닫기">×</button></div>
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
  <div class="bulk-selection" class:selecting>
    <button class="new-group" onclick={() => { selecting = !selecting; selectedIds = []; sessionMenuId = ''; }} disabled={deleting}>{selecting ? '선택 취소' : '대화 선택 삭제'}</button>
    {#if selecting}
      <button class="new-group" onclick={() => selectedIds = selectable.map(s => s.id)} disabled={deleting || !selectable.length} title="모든 폴더·페이지의 대화 선택 (생성 중 제외)">전체 선택</button>
      <button class="new-group" onclick={() => selectedIds = []} disabled={deleting || !selectedIds.length}>선택 해제</button>
      <button class="new-group danger" onclick={deleteSelected} disabled={deleting || !selectedIds.length}>{deleting ? '삭제 중…' : `선택 삭제 (${selectedIds.length})`}</button>
      <small>전체 선택은 모든 폴더·페이지에 적용됩니다. 생성 중 대화는 제외합니다.</small>
    {/if}
  </div>
  <nav>
    <section class="folder-section">
      <button class="folder-section-toggle" onclick={onToggleFolders} aria-expanded={!foldersCollapsed} aria-controls="sidebar-folder-list">
        <span>{foldersCollapsed ? '▸' : '▾'} 폴더</span><small>{groups.length}</small>
      </button>
      {#if !foldersCollapsed}
        <div id="sidebar-folder-list" class="folder-list">
          {#each groups as group, groupIndex}
            <section class="chat-group">
              <div class="chat-group-header">
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
              {#if !collapsedGroups[group.id] && groupPages[group.id]}<SessionPager value={groupPages[group.id]} label={group.name} onPage={(page) => changePage(group.id, page)} />{/if}
              </div>
              {#if !collapsedGroups[group.id]}
                {#each (groupPages[group.id]?.items || []) as session (session.id)}
                  <div class="session-row" class:active={session.id === activeId} class:generating={Boolean(sessionRuns[session.id])}>
                    {#if selecting}<input type="checkbox" aria-label={`${session.title} 선택`} checked={selectedIds.includes(session.id)} disabled={deleting || Boolean(sessionRuns[session.id])} onchange={() => toggleSelection(session.id)} />{/if}
                    <button class="session-select" disabled={deleting} onclick={() => selecting ? toggleSelection(session.id) : onSelect(session.id)}>{session.title}</button>
                    {#if sessionRuns[session.id]}<span class="session-running" title="답변 생성 중" aria-label="답변 생성 중">●</span>{/if}
                    <button class="session-more" disabled={selecting} onclick={() => toggleSessionMenu(session.id)} aria-label={`${session.title} 메뉴`} aria-haspopup="menu" aria-expanded={sessionMenuId === session.id}>⋯</button>
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
          {#if !groups.length}<small class="folder-list-empty">만든 폴더가 없습니다.</small>{/if}
        </div>
      {/if}
    </section>
    <section class="chat-group ungrouped">
      <div class="chat-group-header">
      <button class="group-toggle" onclick={() => onToggleGroup('__ungrouped__')} aria-expanded={!collapsedGroups.__ungrouped__}>
        <span>{collapsedGroups.__ungrouped__ ? '▸' : '▾'} 대화</span><small>{ungroupedSessions.length}</small>
      </button>
      {#if !collapsedGroups.__ungrouped__}<SessionPager value={ungroupedPage} label="미분류 대화" onPage={(page) => changePage('__ungrouped__', page)} />{/if}
      </div>
      {#if !collapsedGroups.__ungrouped__}
        {#each ungroupedPage.items as session (session.id)}
          <div class="session-row" class:active={session.id === activeId} class:generating={Boolean(sessionRuns[session.id])}>
            {#if selecting}<input type="checkbox" aria-label={`${session.title} 선택`} checked={selectedIds.includes(session.id)} disabled={deleting || Boolean(sessionRuns[session.id])} onchange={() => toggleSelection(session.id)} />{/if}
                    <button class="session-select" disabled={deleting} onclick={() => selecting ? toggleSelection(session.id) : onSelect(session.id)}>{session.title}</button>
            {#if sessionRuns[session.id]}<span class="session-running" title="답변 생성 중" aria-label="답변 생성 중">●</span>{/if}
            <button class="session-more" disabled={selecting} onclick={() => toggleSessionMenu(session.id)} aria-label={`${session.title} 메뉴`} aria-haspopup="menu" aria-expanded={sessionMenuId === session.id}>⋯</button>
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
    <button class="library-button" class:active={libraryOpen} onclick={onOpenLibrary}>▤ 라이브러리</button>
    <button class="settings-button" onclick={onOpenSettings}>⚙ 설정</button>
  </div>
  <button class="resize-handle" onpointerdown={onStartResize} aria-label="사이드바 폭 조절"></button>
</aside>
<button class="sidebar-backdrop" onclick={onclose} aria-label="사이드바 닫기"></button>

<style>
.bulk-selection { display: grid; grid-template-columns: minmax(0, 1fr); gap: 5px; }
.bulk-selection.selecting { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.bulk-selection button { text-align: center; min-width: 0; }
.bulk-selection button:disabled { opacity: .4; cursor: default; }
.bulk-selection button:focus-visible { outline: 2px solid #789bff; outline-offset: 2px; }
.bulk-selection button.danger:not(:disabled) { color: #df858b; border-color: #b15d674d; background: #b15d6712; }
.bulk-selection button.danger:not(:disabled):hover { background: #b15d6726; border-color: #b15d6780; }
.bulk-selection small { grid-column: 1 / -1; padding: 3px 2px 0; color: #858e9e; font-size: 11px; line-height: 1.5; }
.session-row input[type="checkbox"] { flex: 0 0 auto; width: 15px; height: 15px; margin: 0 5px; accent-color: #789bff; cursor: pointer; }
</style>
