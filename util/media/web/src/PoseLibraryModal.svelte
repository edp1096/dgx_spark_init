<script>
  import { onMount, onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  export let selectedID = ''
  export let onSelect = () => {}
  export let onClose = () => {}

  let poses = []
  let loading = true
  let loadError = ''
  let category = ''
  let view = ''
  let query = ''
  let visiblePoses = []
  let releaseScroll = null

  const sourceURL = 'https://www.reddit.com/r/comfyui/comments/1v79la4/120_krea2_pose_prompts/'
  const categories = [
    ['standing', '서기'], ['seated on floor', '바닥에 앉기'], ['prone', '엎드리기'],
    ['lying on back', '바로 눕기'], ['squatting', '쪼그리기'], ['kneeling', '무릎 꿇기'],
    ['crawling', '기어가기'], ['fetal', '웅크리기']
  ]
  const views = [
    ['front view', '정면'], ['profile view', '측면'], ['rear view', '후면'],
    ['overhead view', '수직 위'], ['elevated view', '높은 시점']
  ]

  onMount(async () => {
    try {
      const response = await fetch('/pose-library/poses.json')
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      poses = await response.json()
    } catch (error) {
      loadError = `포즈 자료를 불러오지 못했습니다: ${error.message}`
    } finally {
      loading = false
    }
  })

  function filterPoses(allPoses, selectedCategory, selectedView, searchQuery) {
    const needle = searchQuery.trim().toLowerCase()
    return allPoses.filter((pose) => {
      if (selectedCategory && pose.category !== selectedCategory) return false
      if (selectedView && pose.view !== selectedView) return false
      if (needle && !`${pose.name} ${pose.prompt}`.toLowerCase().includes(needle)) return false
      return true
    })
  }

  $: visiblePoses = filterPoses(poses, category, view, query)

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

  function choosePose(pose) {
    onSelect(pose)
    onClose()
  }

  function handleKeydown(event) {
    if (open && event.key === 'Escape') onClose()
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
  <div class="pose-modal-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div class="pose-modal" role="dialog" aria-modal="true" aria-label="Krea 2 포즈 라이브러리">
      <header>
        <div>
          <strong>Krea 2 포즈 라이브러리</strong>
          <small>120개 미리보기에서 원하는 단일 인물 포즈를 고릅니다.</small>
        </div>
        <div class="pose-modal-header-actions">
          <a href={sourceURL} target="_blank" rel="noreferrer">출처 ↗</a>
          <button type="button" aria-label="닫기" onclick={onClose}>×</button>
        </div>
      </header>

      <div class="pose-filters">
        <label>자세
          <select bind:value={category}>
            <option value="">전체 자세</option>
            {#each categories as item}<option value={item[0]}>{item[1]}</option>{/each}
          </select>
        </label>
        <label>시점
          <select bind:value={view}>
            <option value="">전체 시점</option>
            {#each views as item}<option value={item[0]}>{item[1]}</option>{/each}
          </select>
        </label>
        <label class="pose-search">검색
          <input bind:value={query} type="search" placeholder="standing, one knee, overhead…">
        </label>
      </div>

      <div class="pose-result-heading">
        <span>{visiblePoses.length}개 포즈</span>
        <small>화면 비율과 다른 카메라 설정에 따라 결과가 달라질 수 있습니다.</small>
      </div>

      <div class="pose-gallery">
        {#if loading}
          <p class="pose-message">포즈 자료를 불러오는 중…</p>
        {:else if loadError}
          <p class="pose-message error">{loadError}</p>
        {:else if !visiblePoses.length}
          <p class="pose-message">조건에 맞는 포즈가 없습니다.</p>
        {:else}
          {#each visiblePoses as pose}
            <button type="button" class:selected={pose.id === selectedID} onclick={() => choosePose(pose)} title={pose.prompt}>
              <img src={`/pose-library/images/${pose.image}`} alt={pose.name} loading="lazy">
              <span>{pose.name.replace(/^\d+\s*\|\s*/, '')}</span>
              <small>{pose.prompt}</small>
            </button>
          {/each}
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .pose-modal-backdrop {
    position: fixed;
    z-index: 40;
    inset: 0;
    display: grid;
    place-items: center;
    padding: 20px;
    background: #050708d9;
    backdrop-filter: blur(8px);
    overscroll-behavior: contain;
  }

  .pose-modal {
    display: grid;
    grid-template-rows: auto auto auto minmax(0, 1fr);
    width: min(1120px, 96vw);
    height: min(860px, 92vh);
    overflow: hidden;
    overscroll-behavior: contain;
    border: 1px solid #4a555d;
    border-radius: 14px;
    background: #11161a;
    box-shadow: 0 24px 80px #000b;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 14px 16px;
    border-bottom: 1px solid #2d343a;
  }

  header > div:first-child { display: grid; gap: 3px; }
  header strong { color: #e3e8eb; font-size: 14px; }
  header small { color: #76818a; font-size: 10px; }

  .pose-modal-header-actions { display: flex; align-items: center; gap: 10px; }
  .pose-modal-header-actions a { color: #a8d970; font-size: 10px; text-decoration: none; }
  .pose-modal-header-actions button {
    width: 34px;
    height: 34px;
    padding: 0;
    border: 1px solid #3a4248;
    border-radius: 50%;
    color: #c9d0d5;
    background: #1b2126;
    font-size: 20px;
  }

  .pose-filters {
    display: grid;
    grid-template-columns: 170px 170px minmax(220px, 1fr);
    gap: 10px;
    padding: 12px 16px;
    border-bottom: 1px solid #252c31;
    background: #0e1215;
  }

  .pose-filters label { display: grid; gap: 5px; color: #89939a; font-size: 9px; font-weight: 700; }
  .pose-filters select,
  .pose-filters input {
    box-sizing: border-box;
    width: 100%;
    height: 38px;
    padding: 0 10px;
    border: 1px solid #343c42;
    border-radius: 8px;
    color: #d2d8dc;
    background: #171c20;
    font-size: 10px;
  }

  .pose-result-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 9px 16px;
    color: #b8c1c7;
    font-size: 10px;
  }
  .pose-result-heading small { color: #657078; font-size: 9px; text-align: right; }

  .pose-gallery {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    grid-auto-rows: max-content;
    align-content: start;
    align-items: start;
    gap: 9px;
    min-height: 0;
    overflow-y: auto;
    padding: 0 14px 16px;
  }

  .pose-gallery > button {
    display: grid;
    grid-template-rows: auto auto auto;
    align-self: start;
    height: max-content;
    min-width: 0;
    overflow: hidden;
    padding: 0 0 8px;
    border: 1px solid #30383e;
    border-radius: 9px;
    color: #bfc7cc;
    background: #171c20;
    text-align: left;
  }

  .pose-gallery > button:hover { border-color: #687c57; transform: translateY(-1px); }
  .pose-gallery > button.selected { border-color: #a8d970; box-shadow: 0 0 0 2px #a8d97024; }
  .pose-gallery img { width: 100%; aspect-ratio: 1 / 1.28; object-fit: cover; object-position: top; background: #e9e9e6; }
  .pose-gallery span { overflow: hidden; padding: 7px 7px 2px; font-size: 9px; font-weight: 750; text-overflow: ellipsis; white-space: nowrap; }
  .pose-gallery small { overflow: hidden; padding: 0 7px; color: #717b82; font-size: 8px; text-overflow: ellipsis; white-space: nowrap; }
  .pose-message { grid-column: 1 / -1; margin: 50px 0; color: #758089; font-size: 11px; text-align: center; }
  .pose-message.error { color: #d88c85; }

  @media (max-width: 900px) {
    .pose-gallery { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  }

  @media (max-width: 600px) {
    .pose-modal-backdrop { padding: 0; }
    .pose-modal { width: 100vw; height: 100dvh; border: 0; border-radius: 0; }
    header { padding: 12px; }
    header small { max-width: 220px; }
    .pose-filters { grid-template-columns: 1fr 1fr; padding: 10px 12px; }
    .pose-search { grid-column: 1 / -1; }
    .pose-result-heading { align-items: flex-start; padding-inline: 12px; }
    .pose-gallery { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; padding: 0 10px 14px; }
    .pose-gallery span { font-size: 8px; }
  }
</style>
