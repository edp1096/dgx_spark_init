<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'
  import LocalMediaPlayer from './LocalMediaPlayer.svelte'

  export let result = null
  export let onClose = () => {}
  export let onSelectFrames = () => {}
  export let onUpscale = () => {}
  export let onRegenerate = () => {}

  let releaseScroll = null
  let audioElement

  function stopAudio() {
    if (!audioElement) return
    try {
      audioElement.pause()
      audioElement.removeAttribute('src')
      audioElement.load()
    } catch {}
    audioElement = null
  }

  function closeModal() {
    stopAudio()
    onClose()
  }

  function unlockScroll() {
    releaseScroll?.()
    releaseScroll = null
  }

  $: {
    if (result && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!result) unlockScroll()
  }

  onDestroy(() => {
    stopAudio()
    unlockScroll()
  })
</script>

<svelte:window onkeydown={(event) => { if (result && event.key === 'Escape') closeModal() }} />

{#if result}
  <div class="subtitle-modal-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) closeModal() }}>
    <div class="subtitle-modal" role="dialog" aria-modal="true" aria-label="자막 결과 크게 보기">
      <header><div><strong>자막 결과</strong><small title={result.detail}>{result.detail}</small></div><button type="button" aria-label="닫기" onclick={closeModal}>×</button></header>
      <div class="subtitle-modal-content">
        {#if result.mediaSrc}
          <div class="subtitle-modal-player">
            {#if result.audio}
              <audio bind:this={audioElement} controls preload="metadata" src={result.mediaSrc}></audio>
            {:else}
              <LocalMediaPlayer src={result.mediaSrc} autoplay={false} captionSrc={result.captionSrc} captionLabel={result.captionLabel} />
            {/if}
          </div>
        {/if}
        <div class="subtitle-modal-details" class:transcript-only={!result.prompt}>
          <section class="subtitle-modal-transcript"><strong>자막·스크립트</strong><p>{result.transcript || '저장된 자막 본문이 없습니다.'}</p></section>
          {#if result.prompt}<section class="subtitle-modal-source"><strong>원본</strong><p>{result.prompt}</p></section>{/if}
        </div>
      </div>
      <footer>
        <div>{#each result.outputs as output}<a href={output.url} target="_blank" rel="noreferrer">{output.label} ↗</a>{/each}</div>
        <button type="button" onclick={() => onRegenerate(result.jobID)}>자막 재생성</button>
        {#if result.canSelectFrames}<button type="button" onclick={() => onSelectFrames(result.jobID)}>장면 선택</button><button type="button" onclick={() => onUpscale(result.jobID)}>업스케일</button>{/if}
        <button type="button" onclick={closeModal}>닫기</button>
      </footer>
    </div>
  </div>
{/if}

<style>
  .subtitle-modal-backdrop { position:fixed; z-index:100; inset:0; display:grid; place-items:center; padding:20px; background:#050705df; backdrop-filter:blur(8px); overscroll-behavior:contain; }
  .subtitle-modal { display:grid; grid-template-rows:auto minmax(0,1fr) auto; overflow:hidden; width:min(1120px,96vw); max-height:94vh; border:1px solid #3b463c; border-radius:14px; background:#151a16; box-shadow:0 24px 80px #000b; }
  header { position:static; display:flex; align-items:center; justify-content:space-between; width:100%; height:auto; min-height:58px; padding:12px 16px; border-bottom:1px solid #303731; background:#181e19; }
  header div { display:grid; gap:3px; min-width:0; }
  header strong { color:#edf2eb; font-size:14px; }
  header small { overflow:hidden; max-width:min(760px,70vw); color:#7d877e; font-size:10px; text-overflow:ellipsis; white-space:nowrap; }
  header button { margin-left:auto; border:0; color:#aeb7af; background:transparent; font-size:24px; cursor:pointer; }
  .subtitle-modal-content { display:grid; grid-auto-rows:max-content; align-content:start; min-height:0; gap:8px; overflow:auto; padding:8px; background:#090b0a; overscroll-behavior:contain; }
  .subtitle-modal-player { display:grid; place-items:center; overflow:hidden; height:min(calc(94vh - 240px),54vw); min-height:220px; border-radius:9px; background:#000; }
  .subtitle-modal-player:has(audio) { height:auto; min-height:max-content; }
  audio { width:100%; min-height:48px; }
  .subtitle-modal-details { display:grid; grid-template-columns:minmax(0,2fr) minmax(0,1fr); gap:8px; }
  .subtitle-modal-details.transcript-only { grid-template-columns:1fr; }
  .subtitle-modal-transcript,.subtitle-modal-source { display:grid; grid-template-rows:auto minmax(0,1fr); overflow:hidden; height:110px; border:1px solid #2b332c; border-radius:8px; padding:9px 10px; background:#111612; }
  section strong { color:#aeb8af; font-size:10px; }
  section p { overflow:auto; min-height:0; margin:7px 0 0; color:#dce3dd; font-size:11px; line-height:1.65; white-space:pre-wrap; }
  .subtitle-modal-source p { color:#929b93; overflow-wrap:anywhere; }
  footer { display:flex; align-items:center; gap:8px; padding:8px 12px; border-top:1px solid #303731; background:#181e19; }
  footer div { display:flex; flex-wrap:wrap; gap:6px; margin-right:auto; }
  footer a,footer button { display:inline-flex; align-items:center; justify-content:center; min-height:28px; border:1px solid #3c463e; border-radius:6px; padding:4px 9px; color:#cdecaa; background:#202621; font-size:10px; line-height:1; text-decoration:none; cursor:pointer; }
  footer button { color:#b9c4ba; }
  footer a:hover,footer button:hover { border-color:#60705f; background:#293129; }
  @media(max-width:700px) { .subtitle-modal-backdrop{padding:6px}.subtitle-modal-content{padding:7px}.subtitle-modal-details{grid-template-columns:1fr}.subtitle-modal-transcript,.subtitle-modal-source{height:110px}.subtitle-modal-player{height:min(46dvh,56.25vw);min-height:190px} }
</style>
