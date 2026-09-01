<script>
  import { onMount } from 'svelte';
  import Avatar from './Avatar.svelte';

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
  export let onStartResize = () => {};

  let sessionMenuId = '';

  onMount(() => {
    function closeOnOutsidePointer(event) {
      if (sessionMenuId && !event.target.closest?.('.session-menu, .session-more')) sessionMenuId = '';
    }

    function closeOnEscape(event) {
      if (sessionMenuId && event.key === 'Escape') sessionMenuId = '';
    }

    document.addEventListener('pointerdown', closeOnOutsidePointer, true);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('pointerdown', closeOnOutsidePointer, true);
      document.removeEventListener('keydown', closeOnEscape);
    };
  });

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
  <button class="settings-button" onclick={onOpenSettings}>⚙ 설정</button>
  <button class="resize-handle" onpointerdown={onStartResize} aria-label="사이드바 폭 조절"></button>
</aside>
<button class="sidebar-backdrop" onclick={onclose} aria-label="사이드바 닫기"></button>
