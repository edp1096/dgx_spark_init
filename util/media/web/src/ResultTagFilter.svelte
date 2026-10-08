<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'
  import { tagKey } from './lib/jobLists.js'

  export let tags = []
  export let selected = []
  export let excluded = []
  export let untaggedOnly = false
  export let mode = 'or'
  export let label = '결과'
  export let onChange = () => {}
  export let onExcludeChange = () => {}
  export let onUntaggedOnlyChange = () => {}
  export let onModeChange = () => {}

  let open = false
  let query = ''
  let draftSelected = []
  let draftExcluded = []
  let draftUntaggedOnly = false
  let releaseScroll = null

  $: visibleTags = tags.filter((tag) => !query.trim() || tag.name.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()))
  $: draftSelectedKeys = new Set(draftSelected.map(tagKey))
  $: draftExcludedKeys = new Set(draftExcluded.map(tagKey))
  $: filterCount = selected.length + excluded.length + (untaggedOnly ? 1 : 0)

  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    if (!open && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => releaseScroll?.())

  function close() {
    open = false
    query = ''
  }

  function openFilter() {
    draftSelected = [...selected]
    draftExcluded = [...excluded]
    draftUntaggedOnly = untaggedOnly
    open = true
  }

  function removeSelected(name) {
    const key = tagKey(name)
    onChange(selected.filter((tag) => tagKey(tag) !== key))
  }

  function removeExcluded(name) {
    const key = tagKey(name)
    onExcludeChange(excluded.filter((tag) => tagKey(tag) !== key))
  }

  function toggleDraft(name, target) {
    const key = tagKey(name)
    draftUntaggedOnly = false
    if (target === 'include') {
      draftSelected = draftSelectedKeys.has(key) ? draftSelected.filter((tag) => tagKey(tag) !== key) : [...draftSelected, name]
      draftExcluded = draftExcluded.filter((tag) => tagKey(tag) !== key)
    } else {
      draftExcluded = draftExcludedKeys.has(key) ? draftExcluded.filter((tag) => tagKey(tag) !== key) : [...draftExcluded, name]
      draftSelected = draftSelected.filter((tag) => tagKey(tag) !== key)
    }
  }

  function toggleDraftUntagged() {
    draftUntaggedOnly = !draftUntaggedOnly
    if (draftUntaggedOnly) {
      draftSelected = []
      draftExcluded = []
    }
  }

  function clearDraft() {
    draftSelected = []
    draftExcluded = []
    draftUntaggedOnly = false
  }

  function clearAll() {
    onChange([])
    onExcludeChange([])
    onUntaggedOnlyChange(false)
  }

  function apply() {
    onChange(draftSelected)
    onExcludeChange(draftExcluded)
    onUntaggedOnlyChange(draftUntaggedOnly)
    close()
  }
</script>

<svelte:window onkeydown={(event) => { if (open && event.key === 'Escape') close() }} />

<div class="result-tag-filter" aria-label={`${label} 태그 필터`}>
  <button type="button" class="tag-filter-open" class:active={filterCount > 0} onclick={openFilter}>
    태그 필터{#if filterCount}<b>{filterCount}</b>{/if}
  </button>
  {#if filterCount}
    <div class="tag-filter-selected">
      {#if untaggedOnly}<button type="button" class="untagged" title="태그 없음만 필터 제거" onclick={() => onUntaggedOnlyChange(false)}>태그 없음<span>×</span></button>{/if}
      {#each selected as tag}<button type="button" title={`${tag} 포함 필터 제거`} onclick={() => removeSelected(tag)}>#{tag}<span>×</span></button>{/each}
      {#each excluded as tag}<button type="button" class="excluded" title={`${tag} 제외 필터 제거`} onclick={() => removeExcluded(tag)}>제외 #{tag}<span>×</span></button>{/each}
    </div>
    {#if selected.length > 1}
      <div class="tag-match-mode" aria-label="태그 결합 방식">
        <button type="button" class:active={mode === 'or'} onclick={() => onModeChange('or')}>OR</button>
        <button type="button" class:active={mode === 'and'} onclick={() => onModeChange('and')}>AND</button>
      </div>
    {/if}
    <button type="button" class="tag-filter-clear" onclick={clearAll}>초기화</button>
  {:else}
    <small>{tags.length ? `${tags.length}개 태그에서 포함·제외 선택` : '태그 없음 항목을 필터링할 수 있습니다.'}</small>
  {/if}
</div>

{#if open}
  <div class="tag-modal-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) close() }}>
    <section class="tag-filter-modal" aria-label={`${label} 태그 필터 선택`}>
      <header><div><strong>태그 필터</strong><small>{label} · 포함, 제외 또는 태그 없음만 선택합니다.</small></div><button type="button" aria-label="닫기" onclick={close}>×</button></header>
      <div class="tag-filter-search"><input type="search" bind:value={query} placeholder="태그 검색"></div>
      <button type="button" class="tag-untagged-option" class:active={draftUntaggedOnly} onclick={toggleDraftUntagged}><span><strong>태그 없음만</strong><small>태그가 하나도 없는 결과만 표시</small></span><i>{draftUntaggedOnly ? '선택됨' : '선택'}</i></button>
      <div class="tag-option-list">
        {#each visibleTags as tag}
          <div class="tag-option-row">
            <span title={tag.name}>#{tag.name}<small>{tag.count}개</small></span>
            <button type="button" class:active={draftSelectedKeys.has(tagKey(tag.name))} onclick={() => toggleDraft(tag.name, 'include')}>포함</button>
            <button type="button" class="exclude" class:active={draftExcludedKeys.has(tagKey(tag.name))} onclick={() => toggleDraft(tag.name, 'exclude')}>제외</button>
          </div>
        {:else}
          <p>{tags.length ? '일치하는 태그가 없습니다.' : '먼저 결과 항목에 태그를 추가하세요.'}</p>
        {/each}
      </div>
      <footer><button type="button" disabled={!draftSelected.length && !draftExcluded.length && !draftUntaggedOnly} onclick={clearDraft}>모두 해제</button><button type="button" class="primary" onclick={apply}>적용</button></footer>
    </section>
  </div>
{/if}
