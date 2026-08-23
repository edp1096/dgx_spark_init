<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  export let examples = []
  export let selectedID = ''
  export let officialSource = ''
  export let communitySource = ''
  export let onApply = () => {}
  export let onClose = () => {}

  let source = ''
  let category = ''
  let query = ''
  let focusedExample = null
  let visibleExamples = []
  let releaseScroll = null

  const categories = [
    ['photo', '사진'], ['portrait', '인물'], ['anime', '애니'], ['illustration', '일러스트'],
    ['graphic', '그래픽'], ['animal', '동물'], ['fantasy', '판타지·SF'], ['3d', '3D']
  ]

  const categoryByID = {
    'official-rocket': 'photo', 'official-designer-toy': '3d', 'official-collage': 'graphic',
    'official-anime-portrait': 'anime', 'official-ocean': 'illustration', 'official-tree-dog': 'illustration',
    'official-flowers': 'portrait', 'official-mouse': 'animal', 'official-sailor': 'anime',
    'official-coastal-road': 'illustration', 'official-guardian': 'fantasy', 'official-jungle': 'illustration',
    'official-retro-future': 'fantasy', 'official-gold-face': 'portrait', 'official-jester': 'fantasy',
    'official-fashion-red': 'portrait', 'official-ink-faces': 'illustration', 'official-cel-crowd': 'anime',
    'official-wind': 'anime', 'official-film-face': 'portrait', expression: 'portrait', horror: 'photo',
    diversity: 'photo', action: 'anime'
  }

  function sourceType(example) {
    return example.id.startsWith('official-') ? 'official' : 'community'
  }

  function sourceLabel(example) {
    return sourceType(example) === 'official' ? '출처 1 · Krea 공식' : '출처 2 · Sogni'
  }

  function sourceURL(example) {
    return sourceType(example) === 'official' ? officialSource : communitySource
  }

  function filterExamples(allExamples, selectedSource, selectedCategory, searchQuery) {
    const needle = searchQuery.trim().toLowerCase()
    return allExamples.filter((example) => {
      if (selectedSource && sourceType(example) !== selectedSource) return false
      if (selectedCategory && categoryByID[example.id] !== selectedCategory) return false
      if (needle && !`${example.label} ${example.prompt}`.toLowerCase().includes(needle)) return false
      return true
    })
  }

  $: visibleExamples = filterExamples(examples, source, category, query)
  $: if (!open) focusedExample = null

  $: {
    if (open && !releaseScroll) {
      releaseScroll = lockModalScroll()
    } else if (!open && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => {
    releaseScroll?.()
  })

  function handleKeydown(event) {
    if (!open || event.key !== 'Escape') return
    if (focusedExample) focusedExample = null
    else onClose()
  }

  function apply(mode) {
    if (!focusedExample) return
    onApply(focusedExample, mode)
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
  <div class="example-modal-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div class="example-modal" role="dialog" aria-modal="true" aria-label="Krea 2 프롬프트 예제">
      <header>
        <div>
          <strong>Krea 2 프롬프트 예제</strong>
          <small>이미지와 전체 프롬프트를 확인한 뒤 새로 쓰거나 기존 내용 뒤에 추가합니다.</small>
        </div>
        <div class="header-actions">
          <a href={officialSource} target="_blank" rel="noreferrer">출처 1 ↗</a>
          <a href={communitySource} target="_blank" rel="noreferrer">출처 2 ↗</a>
          <button type="button" aria-label="닫기" onclick={onClose}>×</button>
        </div>
      </header>

      {#if focusedExample}
        <div class="example-detail">
          <div class="detail-image"><img src={`/prompt-examples/${focusedExample.id}.webp`} alt={focusedExample.label}></div>
          <div class="detail-copy">
            <button type="button" class="back" onclick={() => focusedExample = null}>← 목록으로</button>
            <span class:official={sourceType(focusedExample) === 'official'} class="source-badge">{sourceLabel(focusedExample)}</span>
            <h3>{focusedExample.label}</h3>
            <p>{focusedExample.prompt}</p>
            <a href={sourceURL(focusedExample)} target="_blank" rel="noreferrer">원문 출처 ↗</a>
            <div class="apply-actions">
              <button type="button" class="secondary" onclick={() => apply('append')}>이어서 적용</button>
              <button type="button" class="primary" onclick={() => apply('replace')}>새로 적용</button>
            </div>
          </div>
        </div>
      {:else}
        <div class="example-filters">
          <label>출처
            <select bind:value={source}><option value="">전체 출처</option><option value="official">출처 1 · Krea 공식</option><option value="community">출처 2 · Sogni</option></select>
          </label>
          <label>종류
            <select bind:value={category}><option value="">전체 종류</option>{#each categories as item}<option value={item[0]}>{item[1]}</option>{/each}</select>
          </label>
          <label class="example-search">검색
            <input bind:value={query} type="search" placeholder="인물, 애니, macro, lighting…">
          </label>
        </div>
        <div class="example-result-heading"><span>{visibleExamples.length}개 예제</span><small>공식 예제가 먼저, 커뮤니티 예제가 다음에 표시됩니다.</small></div>
        <div class="example-gallery">
          {#each visibleExamples as example}
            <button type="button" class:selected={example.id === selectedID} onclick={() => focusedExample = example} title={example.prompt}>
              <img src={`/prompt-examples/${example.id}.webp`} alt={example.label} loading="lazy">
              <span>{example.label}</span>
              <small class:official={sourceType(example) === 'official'}>{sourceType(example) === 'official' ? 'Krea 공식' : 'Sogni'}</small>
            </button>
          {:else}
            <p class="empty">조건에 맞는 예제가 없습니다.</p>
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .example-modal-backdrop { position: fixed; z-index: 45; inset: 0; display: grid; place-items: center; padding: 20px; background: #050708df; backdrop-filter: blur(8px); overscroll-behavior: contain; }
  .example-modal { display: grid; grid-template-rows: auto auto auto minmax(0, 1fr); width: min(1120px, 96vw); height: min(860px, 92vh); overflow: hidden; border: 1px solid #4a555d; border-radius: 14px; background: #11161a; box-shadow: 0 24px 80px #000b; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 14px 16px; border-bottom: 1px solid #2d343a; }
  header > div:first-child { display: grid; gap: 3px; }
  header strong { color: #e3e8eb; font-size: 14px; }
  header small { color: #76818a; font-size: 10px; }
  .header-actions { display: flex; align-items: center; gap: 10px; white-space: nowrap; }
  .header-actions a, .detail-copy > a { color: #a8d970; font-size: 10px; text-decoration: none; }
  .header-actions button { width: 34px; height: 34px; padding: 0; border: 1px solid #3a4248; border-radius: 50%; color: #c9d0d5; background: #1b2126; font-size: 20px; }
  .example-filters { display: grid; grid-template-columns: 190px 170px minmax(220px, 1fr); gap: 10px; padding: 12px 16px; border-bottom: 1px solid #252c31; background: #0e1215; }
  .example-filters label { display: grid; gap: 5px; color: #89939a; font-size: 9px; font-weight: 700; }
  .example-filters select, .example-filters input { box-sizing: border-box; width: 100%; height: 38px; padding: 0 10px; border: 1px solid #343c42; border-radius: 8px; color: #d2d8dc; background: #171c20; font-size: 10px; }
  .example-result-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 9px 16px; color: #b8c1c7; font-size: 10px; }
  .example-result-heading small { color: #657078; font-size: 9px; text-align: right; }
  .example-gallery { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); grid-auto-rows: max-content; align-content: start; gap: 10px; min-height: 0; overflow-y: auto; padding: 0 14px 16px; }
  .example-gallery > button { display: grid; grid-template-rows: auto auto auto; min-width: 0; overflow: hidden; padding: 0 0 9px; border: 1px solid #30383e; border-radius: 10px; color: #c7ced2; background: #171c20; text-align: left; }
  .example-gallery > button:hover { border-color: #687c57; transform: translateY(-1px); }
  .example-gallery > button.selected { border-color: #a8d970; box-shadow: 0 0 0 2px #a8d97024; }
  .example-gallery img { width: 100%; aspect-ratio: 1.3 / 1; object-fit: cover; background: #20262a; }
  .example-gallery span { overflow: hidden; padding: 8px 9px 3px; font-size: 10px; font-weight: 750; text-overflow: ellipsis; white-space: nowrap; }
  .example-gallery small { width: fit-content; margin-left: 9px; padding: 2px 5px; border-radius: 4px; color: #d9a58b; background: #38271f; font-size: 8px; }
  .example-gallery small.official, .source-badge.official { color: #badf91; background: #25331d; }
  .empty { grid-column: 1 / -1; margin: 60px 0; color: #758089; font-size: 11px; text-align: center; }
  .example-detail { grid-row: 2 / -1; display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(340px, .85fr); min-height: 0; overflow: hidden; }
  .detail-image { display: grid; place-items: center; min-height: 0; padding: 18px; background: #090c0e; }
  .detail-image img { display: block; max-width: 100%; max-height: 100%; border-radius: 10px; object-fit: contain; box-shadow: 0 12px 36px #0008; }
  .detail-copy { display: flex; flex-direction: column; align-items: flex-start; gap: 10px; min-height: 0; overflow-y: auto; padding: 22px; border-left: 1px solid #2d343a; }
  .back { padding: 0; border: 0; color: #8d989f; background: transparent; font-size: 10px; }
  .source-badge { padding: 3px 7px; border-radius: 5px; color: #d9a58b; background: #38271f; font-size: 9px; }
  .detail-copy h3 { margin: 2px 0 0; color: #e0e5e8; font-size: 18px; }
  .detail-copy p { margin: 0; color: #b4bdc3; font-size: 11px; line-height: 1.65; white-space: pre-wrap; }
  .apply-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; width: 100%; margin-top: auto; padding-top: 12px; }
  .apply-actions button { min-height: 42px; border-radius: 8px; }
  .apply-actions .secondary { border: 1px solid #3d484f; color: #c5cdd2; background: #1b2227; }

  @media (max-width: 760px) {
    .example-modal-backdrop { padding: 0; }
    .example-modal { width: 100vw; height: 100dvh; border: 0; border-radius: 0; }
    header { padding: 11px 12px; }
    header small { display: none; }
    .header-actions { gap: 7px; }
    .example-filters { grid-template-columns: 1fr 1fr; padding: 10px 12px; }
    .example-search { grid-column: 1 / -1; }
    .example-result-heading { padding-inline: 12px; }
    .example-gallery { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; padding: 0 10px 14px; }
    .example-gallery img { aspect-ratio: 1 / 1; }
    .example-gallery span { padding: 6px 6px 2px; font-size: 8px; }
    .example-gallery small { margin-left: 6px; font-size: 7px; }
    .example-detail { grid-template-columns: 1fr; grid-template-rows: minmax(220px, 43vh) minmax(0, 1fr); overflow-y: auto; }
    .detail-image { padding: 10px; }
    .detail-copy { overflow: visible; padding: 15px; border-top: 1px solid #2d343a; border-left: 0; }
    .detail-copy h3 { font-size: 15px; }
    .apply-actions { position: sticky; bottom: 0; padding: 10px 0 2px; background: #11161a; }
  }
</style>
