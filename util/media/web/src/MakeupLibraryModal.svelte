<script>
  import { onMount, onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  export let selectedID = ''
  export let onSelect = () => {}
  export let onClose = () => {}

  let presets = []
  let loading = true
  let loadError = ''
  let category = ''
  let query = ''
  let visiblePresets = []
  let releaseScroll = null

  const sourceURL = 'https://arca.live/b/aireal/180547516'
  const categories = [
    ['natural', '내추럴'], ['dramatic', '강조'], ['glow', '광채'], ['creative', '크리에이티브'],
    ['editorial', '에디토리얼'], ['regional', '지역 스타일']
  ]

  onMount(async () => {
    try {
      const response = await fetch('/makeup-library/presets.json')
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      presets = await response.json()
    } catch (error) {
      loadError = `메이크업 자료를 불러오지 못했습니다: ${error.message}`
    } finally {
      loading = false
    }
  })

  $: visiblePresets = presets.filter((preset) => {
    const needle = query.trim().toLowerCase()
    if (category && preset.category !== category) return false
    return !needle || `${preset.name} ${preset.prompt}`.toLowerCase().includes(needle)
  })

  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!open && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => releaseScroll?.())

  function choosePreset(preset) {
    onSelect(preset)
    onClose()
  }

  function handleKeydown(event) {
    if (open && event.key === 'Escape') onClose()
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
  <div class="makeup-modal-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div class="makeup-modal" role="dialog" aria-modal="true" aria-label="메이크업과 얼굴 연출 라이브러리">
      <header>
        <div><strong>메이크업·얼굴 연출</strong><small>성별이나 종족과 관계없이 현재 피사체의 얼굴 스타일에 적용합니다.</small></div>
        <div class="header-actions"><a href={sourceURL} target="_blank" rel="noreferrer">출처 ↗</a><button type="button" aria-label="닫기" onclick={onClose}>×</button></div>
      </header>
      <div class="makeup-filters">
        <label>종류<select bind:value={category}><option value="">전체 종류</option>{#each categories as item}<option value={item[0]}>{item[1]}</option>{/each}</select></label>
        <label>검색<input type="search" bind:value={query} placeholder="스모키, 글로시, K-pop…"></label>
      </div>
      <div class="makeup-result-heading"><span>{visiblePresets.length}개 스타일</span><small>동일 시드·구도의 Krea 2 미리보기입니다.</small></div>
      <div class="makeup-gallery">
        {#if loading}<p class="makeup-message">자료를 불러오는 중…</p>
        {:else if loadError}<p class="makeup-message error">{loadError}</p>
        {:else if !visiblePresets.length}<p class="makeup-message">조건에 맞는 스타일이 없습니다.</p>
        {:else}
          {#each visiblePresets as preset}
            <button type="button" class:selected={preset.id === selectedID} onclick={() => choosePreset(preset)} title={preset.prompt}>
              <img src={`/makeup-library/${preset.image}`} alt={preset.name} loading="lazy">
              <span>{preset.name}</span><small>{preset.prompt}</small>
            </button>
          {/each}
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .makeup-modal-backdrop { position: fixed; z-index: 42; inset: 0; display: grid; place-items: center; padding: 20px; background: #050708df; backdrop-filter: blur(8px); overscroll-behavior: contain; }
  .makeup-modal { display: grid; grid-template-rows: auto auto auto minmax(0, 1fr); width: min(1080px, 96vw); height: min(850px, 92vh); overflow: hidden; border: 1px solid #4a555d; border-radius: 14px; background: #11161a; box-shadow: 0 24px 80px #000b; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 14px 16px; border-bottom: 1px solid #2d343a; }
  header > div:first-child { display: grid; gap: 3px; }
  header strong { color: #e3e8eb; font-size: 14px; }
  header small { color: #76818a; font-size: 10px; }
  .header-actions { display: flex; align-items: center; gap: 10px; }
  .header-actions a { color: #a8d970; font-size: 10px; text-decoration: none; }
  .header-actions button { width: 34px; height: 34px; padding: 0; border: 1px solid #394148; border-radius: 8px; color: #aab2b8; background: #191e22; font-size: 19px; }
  .makeup-filters { display: grid; grid-template-columns: 180px minmax(0, 1fr); gap: 10px; padding: 12px 16px; border-bottom: 1px solid #252c31; }
  .makeup-filters label { display: grid; gap: 5px; margin: 0; color: #7f8991; font-size: 9px; }
  .makeup-filters select, .makeup-filters input { height: 36px; padding: 6px 9px; border: 1px solid #343c42; border-radius: 7px; color: #d9dfe2; background: #0d1114; font-size: 10px; }
  .makeup-result-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 9px 16px; color: #9aa49d; font-size: 10px; }
  .makeup-result-heading small { color: #68727a; font-size: 9px; }
  .makeup-gallery { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); grid-auto-rows: max-content; align-content: start; gap: 10px; overflow-y: auto; padding: 0 16px 16px; overscroll-behavior: contain; }
  .makeup-gallery > button { align-self: start; overflow: hidden; min-width: 0; height: auto; padding: 0; border: 1px solid #30383e; border-radius: 10px; color: #aeb6bc; background: #151a1e; text-align: left; }
  .makeup-gallery > button:hover { border-color: #71845f; }
  .makeup-gallery > button.selected { border-color: #9bc76f; box-shadow: 0 0 0 2px #9bc76f44; }
  .makeup-gallery img { display: block; width: 100%; aspect-ratio: 3 / 2; object-fit: cover; object-position: center 38%; }
  .makeup-gallery span, .makeup-gallery small { display: block; overflow: hidden; margin: 0 8px; text-overflow: ellipsis; white-space: nowrap; }
  .makeup-gallery span { margin-top: 8px; color: #d4dadd; font-size: 9px; font-weight: 750; }
  .makeup-gallery small { margin-top: 4px; margin-bottom: 8px; color: #707a81; font-size: 8px; }
  .makeup-message { grid-column: 1 / -1; padding: 50px 10px; color: #768088; font-size: 11px; text-align: center; }
  .makeup-message.error { color: #df8b8b; }
  @media (max-width: 700px) {
    .makeup-modal-backdrop { padding: 0; }
    .makeup-modal { width: 100vw; height: 100dvh; border: 0; border-radius: 0; }
    header { padding: 11px 12px; }
    header small, .makeup-result-heading small { display: none; }
    .makeup-filters { grid-template-columns: 120px minmax(0, 1fr); padding: 9px 10px; }
    .makeup-result-heading { padding: 8px 10px; }
    .makeup-gallery { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; padding: 0 10px 10px; }
  }
</style>
