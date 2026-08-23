<script>
  import { onMount, onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  export let title = '프리셋 이미지 선택'
  export let examples = []
  export let initialTab = 'pose'
  export let onSelect = () => {}
  export let onClose = () => {}

  let poses = []
  let makeup = []
  let tab = initialTab
  let query = ''
  let loading = true
  let loadError = ''
  let releaseScroll = null

  onMount(async () => {
    try {
      const [poseResponse, makeupResponse] = await Promise.all([
        fetch('/pose-library/poses.json'),
        fetch('/makeup-library/presets.json')
      ])
      if (!poseResponse.ok || !makeupResponse.ok) throw new Error(`HTTP ${poseResponse.status}/${makeupResponse.status}`)
      poses = await poseResponse.json()
      makeup = await makeupResponse.json()
    } catch (cause) {
      loadError = `프리셋 이미지를 불러오지 못했습니다: ${cause.message}`
    } finally {
      loading = false
    }
  })

  $: if (open) tab = initialTab || 'pose'
  $: items = tab === 'pose'
    ? poses.map((item) => ({ ...item, library: 'pose', url: `/pose-library/images/${item.image}`, filename: item.image }))
    : tab === 'makeup'
      ? makeup.map((item) => ({ ...item, library: 'makeup', url: `/makeup-library/${item.image}`, filename: item.image }))
      : examples.filter((item) => !/^https?:\/\//.test(item.image || '')).map((item) => ({ ...item, name: item.label, library: 'example', url: `/prompt-examples/${item.image || `${item.id}.webp`}`, filename: item.image || `${item.id}.webp` }))
  $: visibleItems = items.filter((item) => !query.trim() || `${item.name || item.label} ${item.prompt || ''}`.toLowerCase().includes(query.trim().toLowerCase()))

  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!open && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => releaseScroll?.())

  function choose(item) {
    onSelect(item)
  }

  function handleKeydown(event) {
    if (open && event.key === 'Escape') {
      event.stopImmediatePropagation()
      onClose()
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
  <div class="preset-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section class="preset-modal" role="dialog" aria-modal="true" aria-label={title}>
      <header><div><strong>{title}</strong><small>조립기에 준비된 이미지를 참조 입력으로 사용합니다.</small></div><button type="button" aria-label="닫기" onclick={onClose}>×</button></header>
      <div class="preset-toolbar">
        <nav aria-label="프리셋 종류">
          <button type="button" class:active={tab === 'pose'} onclick={() => tab = 'pose'}>포즈 120</button>
          <button type="button" class:active={tab === 'makeup'} onclick={() => tab = 'makeup'}>메이크업 20</button>
          <button type="button" class:active={tab === 'example'} onclick={() => tab = 'example'}>Krea 예제</button>
        </nav>
        <input bind:value={query} type="search" placeholder="이름이나 프롬프트 검색">
      </div>
      <div class="preset-count">{visibleItems.length}개</div>
      <div class="preset-gallery">
        {#if loading}<p>프리셋을 불러오는 중…</p>
        {:else if loadError}<p class="error">{loadError}</p>
        {:else if !visibleItems.length}<p>조건에 맞는 이미지가 없습니다.</p>
        {:else}{#each visibleItems as item}
          <button type="button" onclick={() => choose(item)} title={item.prompt || item.name}>
            <img src={item.url} alt={item.name || item.label} loading="lazy">
            <span>{(item.name || item.label).replace(/^\d+\s*\|\s*/, '')}</span>
          </button>
        {/each}{/if}
      </div>
    </section>
  </div>
{/if}

<style>
  .preset-backdrop { position:fixed; z-index:70; inset:0; display:grid; place-items:center; padding:20px; background:#050708d9; backdrop-filter:blur(8px); overscroll-behavior:contain; }
  .preset-modal { display:grid; grid-template-rows:auto auto auto minmax(0,1fr); width:min(1120px,96vw); height:min(860px,92vh); overflow:hidden; border:1px solid #4a555d; border-radius:14px; background:#11161a; box-shadow:0 24px 80px #000b; }
  header { display:flex; align-items:center; justify-content:space-between; gap:14px; padding:13px 16px; border-bottom:1px solid #2d343a; }
  header > div { display:grid; gap:3px; }
  header strong { color:#e3e8eb; font-size:14px; }
  header small { color:#76818a; font-size:10px; }
  header button { width:34px; height:34px; padding:0; border:1px solid #3a4248; border-radius:50%; color:#c9d0d5; background:#1b2126; font-size:20px; }
  .preset-toolbar { display:grid; grid-template-columns:auto minmax(180px,1fr); gap:12px; padding:12px 16px; border-bottom:1px solid #252c31; background:#0e1215; }
  nav { display:flex; gap:6px; }
  nav button { min-height:36px; padding:7px 12px; border:1px solid #343c42; border-radius:8px; color:#aab3b9; background:#171c20; font-size:10px; }
  nav button.active { border-color:#8ebd5e; color:#dff8c1; background:#25351d; }
  input { box-sizing:border-box; width:100%; height:36px; padding:0 11px; border:1px solid #343c42; border-radius:8px; color:#d2d8dc; background:#171c20; font-size:10px; }
  .preset-count { padding:8px 16px; color:#8f9aa1; font-size:10px; }
  .preset-gallery { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); grid-auto-rows:max-content; align-content:start; align-items:start; gap:9px; min-height:0; overflow-y:auto; padding:0 14px 16px; }
  .preset-gallery > button { display:grid; grid-template-rows:auto auto; align-self:start; height:auto; min-width:0; overflow:hidden; padding:0 0 8px; border:1px solid #30383e; border-radius:9px; color:#cbd2d6; background:#171c20; text-align:left; }
  .preset-gallery img { display:block; width:100%; aspect-ratio:3/4; object-fit:cover; object-position:center top; background:#090c0e; }
  .preset-gallery span { overflow:hidden; padding:7px 8px 0; font-size:9px; line-height:1.35; text-overflow:ellipsis; white-space:nowrap; }
  .preset-gallery p { grid-column:1/-1; color:#8f9aa1; text-align:center; }
  .preset-gallery p.error { color:#e58f8f; }
  @media (max-width:700px) {
    .preset-backdrop { padding:8px; }
    .preset-modal { width:100%; height:94vh; }
    .preset-toolbar { grid-template-columns:1fr; gap:8px; }
    nav { display:grid; grid-template-columns:repeat(3,1fr); }
    nav button { padding-inline:5px; }
    .preset-gallery { grid-template-columns:repeat(3,minmax(0,1fr)); gap:7px; padding-inline:9px; }
  }
</style>
